# data

- `agenteval_traces.jsonl`: the sample traces shipped with [agenteval](https://github.com/AnhDuongVo/agenteval) (OpenAI, LangGraph and LlamaIndex formats). `agentsim run --profile data/agenteval_traces.jsonl` converts them on the fly.
- `measurement_example.jsonl`: an illustrative measurement in the shape `agentsim bench` writes, with numbers of the order one sees for a ~8B model on one data-centre GPU. It is not a measurement of any specific endpoint; run `agentsim bench` against yours and calibrate from that.
