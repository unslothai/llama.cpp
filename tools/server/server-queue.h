#pragma once

#include "server-task.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// struct for managing server tasks
// in most cases, use server_response_reader to post new tasks and retrieve results
struct server_queue {
private:
    int id = 0;
    bool running  = false;
    bool sleeping = false;
    bool req_stop_sleeping = false;
    int64_t time_last_task = 0;

    // queues
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;
    // tasks declined while yielding, put back in queue_tasks once the yield is done
    // note: kept as a member so that cleanup_pending_task() can also reach them
    std::deque<server_task> queue_tasks_unhandled;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // used by yield_to_queue, all fields are guarded by mutex_tasks
    struct worker_t {
        std::thread             thread;
        std::condition_variable cv;        // the worker sleeps on this until a yield starts
        std::exception_ptr      exception; // exception thrown while processing tasks, if any
        bool stop     = false;
        bool busy     = false; // set by yield_to_queue(), cleared by the worker once it is done processing tasks
        bool yielding = false; // work() is still running on the start_loop() thread
    };
    worker_t worker;

    // callback functions
    std::function<bool(server_task &&, bool)> callback_new_task;
    std::function<void(void)>                 callback_update_slots;
    std::vector<std::function<void(bool)>>    callback_sleeping_state;

public:
    ~server_queue() { worker_stop(); }

    // Add a new task to the end of the queue
    int post(server_task && task, bool front = false);

    // multi-task version of post()
    int post(std::vector<server_task> && tasks, bool front = false);

    // Add a new task, but defer until one slot is available
    void defer(server_task && task);

    // Get the next id for creating a new task
    int get_new_id();

    // Call when the state of one slot is changed, it will move one task from deferred to main queue
    // prioritize tasks that use the specified slot (otherwise, pop the first deferred task)
    void pop_deferred_task(int id_slot);

    // if sleeping, request exiting sleep state and wait until it is done
    // returns immediately if not sleeping
    void wait_until_no_sleep();

    bool is_sleeping() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return sleeping;
    }

    // end the start_loop routine
    void terminate();

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     *
     * Sleeping procedure (disabled if idle_sleep_ms < 0):
     * - If there is no task after idle_sleep_ms, enter sleeping state
     *   note: metrics tasks are processed as usual, but do not reset the idle timer
     * - Call callback_sleeping_state(true)
     * - Wait until req_stop_sleeping is set to true
     * - Call callback_sleeping_state(false)
     * - Exit sleeping state
     */
    void start_loop(int64_t idle_sleep_ms = -1);

    // while waiting for work() to finish, run process_new_tasks on the worker thread
    // returns once work() is done (may throw exceptions)
    // must be called from start_loop() thread (ideally inside callback_update_slots)
    // use case: return metrics while encode/decode is running
    // ref: https://github.com/ggml-org/llama.cpp/pull/27041
    //
    // tasks declined by callback_new_task are put back in the queue once this returns
    void yield_to_queue(std::function<void()> && work);

    // for metrics
    size_t queue_tasks_deferred_size() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return queue_tasks_deferred.size();
    }

    //
    // Functions below are not thread-safe, must only be used before start_loop() is called
    //

    // Register function to process a new task
    // the second argument tells whether the queue is currently yielding (see yield_to_queue)
    // only then may the callback return false to decline the task, and it must leave it
    // untouched, so that it can be put back in the queue later
    // note: while yielding, the callback runs on worker thread, not main thread
    void on_new_task(std::function<bool(server_task &&, bool)> callback) {
        callback_new_task = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Register callback for sleeping state change; multiple callbacks are allowed
    // for example: register order cb0, cb1, cb2
    // entering sleep: queue.sleeping = true --> cb0(true) --> cb1(true) --> cb2(true)
    // leaving sleep: cb2(false) --> cb1(false) --> cb0(false) --> queue.sleeping = false
    // note: caller will hold mutex_tasks while calling the callbacks
    void on_sleeping_state(std::function<void(bool)> callback) {
        callback_sleeping_state.push_back(std::move(callback));
    }

private:
    void cleanup_pending_task(int id_target);

    // process all pending tasks in the queue
    // returns true if the queue is terminated, false if there is no more task to process
    // while yielding, declined tasks are moved to queue_tasks_unhandled
    bool process_new_tasks(bool is_yielding);

    // for worker_t
    void worker_loop();
    void worker_stop();
};

// struct for managing server responses
// in most cases, use server_response_reader to retrieve results
struct server_response {
private:
    bool running = true;

    // One waiter per reader, shared by every task id that reader registered in one call.
    // Results are queued on the waiter that owns the id, so sending a result wakes only the
    // thread that is waiting for it, and that thread finds its result without searching.
    //
    // Previously there was a single result vector and a single condition variable: every
    // result woke every waiting HTTP thread, and each of them re-took the mutex and scanned
    // the whole vector before going back to sleep. With N slots generating that is N wakeups
    // and N scans per token, i.e. N^2 per decode step, all of it contending with the decode
    // thread for the same mutex.
    struct waiter {
        std::condition_variable cv;

        // FIFO, so results are handed out in the order they were sent, as before
        std::deque<server_task_result_ptr> results;
    };

    using waiter_ptr = std::shared_ptr<waiter>;

    // task id --> the waiter that is expecting its results
    std::unordered_map<int, waiter_ptr> waiting;

    std::mutex mutex_results;

    // only used to park a reader whose ids are no longer in the waiting list, so that it
    // still returns after the timeout it asked for rather than spinning
    std::condition_variable condition_gone;

    // all ids registered together share one waiter, so the first hit is the right one
    // must be called with mutex_results held
    waiter_ptr find_waiter(const std::unordered_set<int> & id_tasks) const;

public:
    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task);

    void add_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task);

    // remove multiple tasks from waiting list
    void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // This function blocks the thread until there is a response for one of the id_tasks
    server_task_result_ptr recv(const std::unordered_set<int> & id_tasks);

    // same as recv(), but have timeout in seconds
    // if timeout is reached, nullptr is returned
    server_task_result_ptr recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout);

    // single-task version of recv()
    server_task_result_ptr recv(int id_task);

    // Send a new result to a waiting id_task
    void send(server_task_result_ptr && result);

    // broadcast a new result to all waiting tasks
    // (used by router mode)
    void broadcast(server_task_result_ptr && result);

    // terminate the waiting loop
    void terminate();
};

// RAII wrapper to make working with server_queue and server_response easier
// it provides a generator-like API for server responses
// support pooling connection state and aggregating multiple results
struct server_response_reader {
    std::unordered_set<int> id_tasks;
    server_queue & queue_tasks;
    server_response & queue_results;
    size_t received_count = 0;
    bool cancelled = false;
    int polling_interval_seconds;

    // tracking generation state and partial tool calls
    // only used by streaming completions
    std::vector<task_result_state> states;

    // should_stop function will be called each polling_interval_seconds
    server_response_reader(server_queue & queue_tasks, server_response & queue_results, int polling_interval_seconds)
        : queue_tasks(queue_tasks), queue_results(queue_results), polling_interval_seconds(polling_interval_seconds) {}
    ~server_response_reader() {
        stop();
    }

    int get_new_id() {
        return queue_tasks.get_new_id();
    }

    // if front = true, the task will be posted to the front of the queue (high priority)
    void post_task(server_task && task, bool front = false);
    void post_tasks(std::vector<server_task> && tasks, bool front = false);
    bool has_next() const;

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop);

    struct batch_response {
        bool is_terminated = false; // if true, indicates that processing was stopped before all results were received
        std::vector<server_task_result_ptr> results;
        server_task_result_ptr error; // nullptr if no error
    };
    // aggregate multiple results
    batch_response wait_for_all(const std::function<bool()> & should_stop);

    void stop();
};
