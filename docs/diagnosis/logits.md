# Phase 3 — top-10 logprobs comparison (vLLM vs llama-server)

prompt: `What is the capital of France? Answer in exactly one word.`  
temperature: 0.0 (greedy), max_tokens: 1, top_logprobs: 10

vLLM completion: `Paris`  
llama-server completion: `Paris`

## Top-10 next-token logprobs

| rank | vLLM token | vLLM logprob | llama token | llama logprob |
| --- | --- | --- | --- | --- |
| 1 | `token_id:42572` | -0.0001 | `Paris` | -0.0000 |
| 2 | `token_id:6993` | -9.8751 | ` Paris` | -11.4609 |
| 3 | `token_id:1784` | -12.0001 | `Par` | -12.5013 |
| 4 | `token_id:2029` | -12.9376 | `PAR` | -14.4139 |
| 5 | `token_id:3814` | -13.2501 | `巴黎` | -14.6281 |
| 6 | `token_id:102726` | -13.6251 | `Pars` | -15.6960 |
| 7 | `token_id:38166` | -13.6251 | `par` | -16.1068 |
| 8 | `token_id:72056` | -13.8751 | ` Париж` | -16.8023 |
| 9 | `token_id:75613` | -15.1876 | `Berlin` | -17.0093 |
| 10 | `token_id:126441` | -15.3751 | ` París` | -17.9099 |

Jaccard(top-10 token set): **0.00**  
Top-1 match: **False**  

## TL;DR

Greedy top-1 token: **DIFFERS**.
