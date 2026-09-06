# Four distinct prompts, each about 300 tokens of raw text (no chat template).
_BODIES = {
"P0": """The history of numerical computing is a history of compromises between speed and exactness.
Early machines used fixed point arithmetic because it was cheap, and programmers carried scaling
factors in their heads. Floating point hardware moved the bookkeeping into silicon, but it did not
remove the compromise, it only hid it. Addition of floating point numbers is commutative but it is
not associative, so the order in which a long sum is accumulated changes the last few bits of the
result. On a single processor that order is fixed by the program text and nobody notices. On a
parallel processor the order is fixed by how the work was divided, and the division is chosen for
speed, not for reproducibility. A reduction split across two warps sums a different set of partial
products than the same reduction split across four warps, and the two answers differ in the low
bits. Nothing is wrong with either answer. Both are within a fraction of an ulp of the exact value.
The trouble begins when a downstream decision is discrete. A comparison, a rounding to an integer,
or the selection of the largest element of a vector turns a difference of one bit into a difference
of one branch, and from there the two computations walk away from each other and never come back.
Explain, carefully and at length, why this matters for a system that serves many users at once,
and what an engineer would have to give up to make the answer depend only on the request and not
on what else the machine happened to be doing at the time. Discuss the cost.""",
"P1": """Consider a public library that lends physical books and must decide how many copies of a
popular title to buy. The librarian has a fixed budget, a waiting list that grows and shrinks, and
a shelf that is already full. Every copy purchased shortens the queue for that title and lengthens
the queue for every other title, because the money and the shelf space are shared. The obvious
policy, buy copies of whatever has the longest queue, is unstable, because a title that briefly
becomes fashionable will absorb the whole budget and then sit unread for a decade. A better policy
has to weigh how long the demand is likely to last against how long the book will remain useful,
and it has to do this with almost no information. Describe in detail how you would design such a
policy, what data you would collect, how you would test it without harming readers, and how you
would know whether it was working. Consider what happens when the budget is cut in half without
warning, when a title is suddenly assigned as required reading by a local school, and when the
shelf itself must shrink because the building is being renovated. Explain the tradeoffs plainly.""",
"P2": """A small coastal town has one bridge to the mainland and it is failing. The engineers say it
has perhaps eight years left. Replacing it costs more than the town has ever spent on anything.
Repairing it buys maybe four years and costs a third as much, and the repair work closes the bridge
for two months in the summer, which is when the town earns most of its money. Doing nothing is
free until the day it is not. The town council is split, the ferry operator has opinions, and the
regional government will match funds only for a replacement, only if construction begins within
three years, and only if the town covers the first quarter of the cost itself. Write a long and
careful analysis of the options available to the council. Identify the assumptions that matter
most, the ones where being wrong changes the recommendation, and say how the council could cheaply
find out whether those assumptions hold. Then give a recommendation and state honestly what would
have to be true for the recommendation to be wrong. Do not hedge. Commit to an answer at the end.""",
"P3": """Describe the process by which a large body of water freezes over in winter, beginning with
the surface layer and working downward, and explain why the ice floats rather than sinking, why a
deep lake takes much longer to freeze than a shallow one of the same surface area, and why the
temperature at the bottom of a frozen lake settles near four degrees Celsius rather than at zero.
Then explain what this means for the animals that live there, how fish survive a winter under a
solid lid, why a heavy snowfall on top of the ice can be more dangerous to them than the cold
itself, and what happens in the spring when the whole column overturns. Use plain language and
avoid equations. Where a common explanation is wrong or incomplete, say so and give the better one.
Be thorough. Assume the reader is curious and patient but has no training in physics or biology,
and would rather understand one thing properly than be told five things quickly.""",
}
PROMPTS = {k: " ".join(v.split()) for k, v in _BODIES.items()}
