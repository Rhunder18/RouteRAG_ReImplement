from routerag.llm.unsloth_offline import UnslothOffline

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .base import BaseLLM
from .gateway_llm import OpenAICompatibleGatewayLLM
from .gateways import GATEWAYS, GatewaySpec, find_gateway

logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    gateway = find_gateway(config.llm_name)
    if gateway is not None:
        return OpenAICompatibleGatewayLLM(config, gateway)
        
    return CacheOpenAI.from_experiment_config(config)
