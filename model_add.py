from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer
from llama_cpp import Llama
import os
import logging
from typing import Dict, List, Union, Optional

logger = logging.getLogger(__name__)

class LlamaCPPInvocationLayer(PromptModelInvocationLayer):
    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        max_length: Optional[int] = 128,
        max_context: Optional[int] = 32000,
        n_parts: Optional[int] = -1,
        seed: Optional[int] = 1337,
        f16_kv: Optional[bool] = True,
        logits_all: Optional[bool] = False,
        vocab_only: Optional[bool] = False,
        use_mmap: Optional[bool] = True,
        use_mlock: Optional[bool] = False,
        embedding: Optional[bool] = False,
        n_threads: Optional[int] = 8,
        n_batch: Optional[int] = 512,
        last_n_tokens_size: Optional[int] = 64,
        lora_base: Optional[str] = None,
        lora_path: Optional[str] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        super().__init__()
        if not model_name_or_path:
            raise ValueError("model_name_or_path must not be empty")

        self.model_name_or_path = model_name_or_path
        self.max_context = max_context
        self.max_length = max_length
        self.verbose = verbose

        # Initialize Llama model
        self.model = Llama(
            model_path=model_name_or_path,
            n_ctx=max_context,
            n_parts=n_parts,
            seed=seed,
            f16_kv=f16_kv,
            logits_all=logits_all,
            vocab_only=vocab_only,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            embedding=embedding,
            n_threads=n_threads,
            n_batch=n_batch,
            last_n_tokens_size=last_n_tokens_size,
            lora_base=lora_base,
            lora_path=lora_path,
            verbose=verbose
        )

    def __del__(self):
        if hasattr(self, "model") and self.model is not None:
            self.model.close()

    def _ensure_token_limit(self, prompt: str) -> str:
        tokenized_prompt = self.model.tokenize(prompt.encode("utf-8"))
        if len(tokenized_prompt) + self.max_length > self.model.n_ctx():
            logger.warning(
                "Prompt truncated from %s to %s tokens to fit model context window (%s max tokens)",
                len(tokenized_prompt),
                max(0, self.model.n_ctx() - self.max_length),
                self.model.n_ctx()
            )
            tokenized_prompt = tokenized_prompt[: max(0, self.model.n_ctx() - self.max_length)]
            return self.model.detokenize(tokenized_prompt).decode("utf-8")
        return prompt

    def invoke(self, *args, **kwargs) -> List[str]:
        prompt = kwargs.get("prompt", "")
        prompt = self._ensure_token_limit(prompt)

        stream = kwargs.pop("stream", False)
        model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "suffix", "max_tokens", "temperature", "top_p", "logprobs", "echo",
                "repeat_penalty", "top_k", "stop"
            ]
            if key in kwargs
        }

        if stream:
            output = []
            for token in self.model(prompt, stream=True, **model_input_kwargs):
                output.append(token["choices"][0]["text"])
            return output
        else:
            output = self.model(prompt, **model_input_kwargs)
            return [choice["text"] for choice in output["choices"]]

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        return bool(model_name_or_path)