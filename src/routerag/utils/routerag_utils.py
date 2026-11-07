SYSTEM_ROLE = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine using the following strict format:
1. You MUST first decide which retrieval mode to use (both modes will return relevant documents, but use different retrieval methods):
   - Use [passage] to find documents using semantic similarity-based dense retrieval
   - Use [graph] to find documents through graph-based retrieval, which performs retrieval on a structured knowledge graph constructed from documents using fact ranking and graph reasoning
   - You can combine them as [graph][passage] to get documents from both retrieval methods
2. Then formulate your specific search query based on what information you need
3. Finally, wrap everything in <search> and </search> tags
For example:
- Using dense retrieval: <search> [passage] the capital of France </search>
- Using graph-based retrieval: <search> [graph] the capital of France </search>
- Using both methods: <search> [graph][passage] the capital of France </search>
The search results (relevant documents) will be returned between <information> and </information> tags. \
You can search as many times as you want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Paris </answer>\n"""

INFO_TEXT = '\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n'