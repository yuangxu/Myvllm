# LLMEngine：对外暴露的顶层推理 API
#
# __init__(model_config, engine_config):
#   self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
#   self.scheduler = Scheduler(engine_config, block_manager)
#   self.model_runner = ModelRunner(model_config, engine_config)
#
# add_request(prompt: str, sampling_params: SamplingParams):
#   token_ids = tokenizer.encode(prompt)
#   seq = Sequence(token_ids, sampling_params)
#   scheduler.add_sequence(seq)
#
# step() -> List[(seq_id, completion_token_ids)]:
#   prefill_seqs, decode_seqs = scheduler.schedule()
#   new_token_ids = model_runner.run(prefill_seqs, decode_seqs)
#   scheduler.postprocess(decode_seqs, new_token_ids)
#   return [(seq.seq_id, seq.completion_token_ids) for seq in finished]
#
# generate(prompts: List[str], sampling_params) -> List[str]:
#   for p in prompts: add_request(p, sampling_params)
#   results = {}
#   while not scheduler.is_finished():
#       finished = step()
#       results.update({seq_id: tokens for seq_id, tokens in finished})
#   return [tokenizer.decode(results[i]) for i in sorted(results)]
