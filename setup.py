import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="routerag",
    version="1.0",
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.1",
        "transformers==4.57.6",
        "openai>=3.3.1,<4",
        "litellm==1.73.1",
        "networkx>=3.4.2",
        "python_igraph>=0.11.8",
        "tiktoken>=0.7.0",
        "pydantic>=2.10.4,<3",
        "tenacity==8.5.0",
        "einops", # No version specified
        "tqdm", # No version specified
        "boto3", # No version specified
        "nest_asyncio",
        "numpy",
        "pandas",
        "pyarrow",
        "requests",
        "scipy",
        "filelock",
        "httpx>=0.27,<1",
    ],
    extras_require={
        "milvus": ["pymilvus[milvus_lite]>=2.4.2"],
        "qdrant": ["qdrant-client>=1.9"],
        "chroma": ["chromadb>=0.5"],
        "transformers-embedding": ["sentence-transformers>=3.0"],
        "gritlm": ["gritlm==1.0.2"],
        "vllm": ["vllm"],
        "mcp": ["mcp>=2.2,<3"],
    },
    entry_points={"console_scripts": ["routerag-mcp=routerag.mcp_server:main"]},
    package_data={"routerag": ["prompts/dspy_prompts/*.json"]},
    include_package_data=True,
)
