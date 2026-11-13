from typing import List
import torch.cuda
import outlines.generate as generate
import outlines.models as models

from .base import LLMConfig
from ..utils.llm_utils import TextChatMessage, get_pydantic_model
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

from unsloth import FastLanguageModel
from transformers import PreTrainedTokenizer

def convert_text_chat_messages_to_strings(messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer) -> List[str]:
    return tokenizer.apply_chat_template(conversation=messages, tokenize=False)

def convert_text_chat_messages_to_input_string(messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer) -> str:
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        chat_template=None,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
        tools=None,
        documents=None,
    )
    return prompt

class UnslothOffline:

    def _init_llm_config(self) -> None:
        self.llm_config = LLMConfig()

    def __init__(self, global_config, cache_dir=None, cache_filename=None, max_model_len=4096, **kwargs):
        model_name = kwargs.get('model_name', global_config.llm_name)
        if model_name is None:
            model_name = 'meta-llama/Llama-3.1-8B-Instruct'
        model_name = model_name.removeprefix("Unsloth/")
        self.model_name = model_name
        if kwargs.get("model") and kwargs.get("tokenizer"):
            self.model = kwargs.get("model")
            self.tokenizer = kwargs.get("tokenizer")
        else:   
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                self.model_name,
                dtype="auto",
                device_map="balanced"
            )
            FastLanguageModel.for_inference(self.model)
        
        if not getattr(self.model.config, "is_encoder_decoder", False):
            if self.tokenizer.padding_side != "left":
                self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        if cache_filename is None:
            cache_filename = f'{model_name.replace("/", "_")}_cache.sqlite'
        if cache_dir is None:
            cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        self.cache_file_name = os.path.join(cache_dir, cache_filename)
    
    def _generate_text(self, prompt_text: str, max_tokens: int) -> str:
        model_inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        generated = self.model.generate(**model_inputs, max_new_tokens=max_tokens, do_sample=False)
        completion_ids = generated[:, model_inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(completion_ids[0], skip_special_tokens=True)

    def infer(self, messages: List[TextChatMessage], max_tokens=2048):
        logger.info(f"Calling Unsloth offline, # of messages {len(messages)}")
        prompt_text = convert_text_chat_messages_to_input_string(messages, self.tokenizer)
        response = self._generate_text(prompt_text, max_tokens)
        prompt_tokens = len(self.tokenizer.encode(prompt_text))
        completion_tokens = len(self.tokenizer.encode(response))
        metadata = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
        return response, metadata

    def batch_infer(self, messages_list: List[List[TextChatMessage]], max_tokens=2048, json_template=None):
        if len(messages_list) > 1:
            logger.info(f"Calling Unsloth offline, # of messages {len(messages_list)}, using batchsize = 4")

        all_prompt_texts = [convert_text_chat_messages_to_input_string(messages, self.tokenizer) for messages in messages_list]

        if json_template is not None:
            guided_json=get_pydantic_model(json_template)
            outlines_model = outlines.from_transformers(self.model, self.tokenizer)
            all_responses = [
                outlines_model(prompt_text, guided_json, max_new_tokens=max_tokens)
                for prompt_text in all_prompt_texts
            ]
        else:
            all_responses = [self._generate_text(prompt_text, max_tokens) for prompt_text in all_prompt_texts]
        if len(all_responses) != len(messages_list):
            raise RuntimeError(f"Transformers offline returned {len(all_responses)} outputs for {len(messages_list)} prompts.")
        all_prompt_tokens = [len(self.tokenizer.encode(prompt)) for prompt in all_prompt_texts]
        all_completion_tokens = [len(self.tokenizer.encode(response)) for response in all_responses]

        metadata = {
            "prompt_tokens": sum(all_prompt_tokens),
            "completion_tokens": sum(all_completion_tokens),
            "num_request": len(messages_list)
        }
        return all_responses, metadata
