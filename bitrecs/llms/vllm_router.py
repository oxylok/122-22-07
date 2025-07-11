from openai import OpenAI

class vLLM:
    """
     python3 -m vllm.entrypoints.openai.api_server --model NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key xxxxxxx

    """
    def __init__(self, 
                 key, 
                 model="NousResearch/Meta-Llama-3-8B-Instruct", 
                 system_prompt="You are a helpful assistant.", 
                 temp=0.0,
                 base_url="http://localhost:8000/v1"):        
        self.VLLM_API_KEY = key
        if not self.VLLM_API_KEY:
            raise ValueError("VLLM_API_KEY is not set")        
        self.model = model        
        self.system_prompt = system_prompt
        self.temp = temp
        self.base_url = base_url
        

    def call_vllm(self, user_prompt) -> str:
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.VLLM_API_KEY,
        )
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temp,
            max_tokens=2048
        )
        #print(completion.choices[0].message.content)
        result = completion.choices[0].message.content
        return result

        
