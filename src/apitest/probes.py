IDENTITY_PROBES = [
    {
        "name": "self_identity",
        "prompt": "What model are you? What company created you? Answer in one sentence.",
        "category": "identity",
    },
    {
        "name": "knowledge_cutoff",
        "prompt": "What is your knowledge cutoff date? What is the latest major world event you are aware of? Keep it brief.",
        "category": "knowledge",
    },
    {
        "name": "reasoning_style",
        "prompt": "Solve step by step: If a shirt costs $30 after a 25% discount, what was the original price? Show your work.",
        "category": "style",
    },
    {
        "name": "coding_style",
        "prompt": "Write a one-line comment describing what programming language and style you use.",
        "category": "style",
    },
    {
        "name": "chinese_identity",
        "prompt": "请用中文回答：你是哪个公司开发的？你是什么模型？",
        "category": "identity",
    },
]

MODEL_FINGERPRINTS = {
    "gpt-4o": {"knowledge_cutoff": ["2024", "2025"], "company": "OpenAI"},
    "gpt-4-turbo": {"knowledge_cutoff": ["2023", "2024"], "company": "OpenAI"},
    "gpt-4o-mini": {"knowledge_cutoff": ["2024", "2025"], "company": "OpenAI"},
    "o1": {"knowledge_cutoff": ["2024", "2025"], "company": "OpenAI"},
    "claude-sonnet-4": {"knowledge_cutoff": ["2025"], "company": "Anthropic"},
    "claude-opus-4": {"knowledge_cutoff": ["2025"], "company": "Anthropic"},
    "claude-haiku": {"knowledge_cutoff": ["2025"], "company": "Anthropic"},
    "deepseek-v3": {"knowledge_cutoff": ["2024", "2025"], "company": "DeepSeek"},
    "gemini": {"knowledge_cutoff": ["2024", "2025"], "company": "Google"},
    "qwen": {"knowledge_cutoff": ["2024", "2025"], "company": "Alibaba"},
}
