// [TAG_EXACT_CONCURRENCY] CUDA must publish a non-blocking event query, or every asynchronous park poll blocks.

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>

int main() {
    ggml_backend_load_all();

    int rc = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        const char * reg_name  = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        const bool   supported = ggml_backend_dev_supports_event_query(dev);

        printf("%s (%s): event query %s\n", ggml_backend_dev_name(dev), reg_name, supported ? "yes" : "no");

        if (!supported) {
            if (strcmp(reg_name, "CUDA") == 0) {
                fprintf(stderr, "%s is a CUDA device without a non-blocking event query\n", ggml_backend_dev_name(dev));
                rc = 1;
            }
            continue;
        }

        ggml_backend_t       backend = ggml_backend_dev_init(dev, nullptr);
        ggml_backend_event_t event   = backend ? ggml_backend_event_new(dev) : nullptr;

        if (!event) {
            fprintf(stderr, "%s reports an event query but gave no event\n", ggml_backend_dev_name(dev));
            rc = 1;
        } else {
            ggml_backend_event_record(event, backend);
            ggml_backend_event_synchronize(event);

            if (!ggml_backend_event_query(event)) {
                fprintf(stderr, "%s says a completed event is still pending\n", ggml_backend_dev_name(dev));
                rc = 1;
            }

            ggml_backend_event_free(event);
        }

        if (backend) {
            ggml_backend_free(backend);
        }
    }

    if (rc == 0) {
        printf("ok\n");
    }

    return rc;
}
