from openai import OpenAI
from openai.types.responses import Response

class ChatGPT:
    def __init__(self, 
                 key,
                 model="gpt-4o-mini", 
                 system_prompt="You are a helpful assistant.", 
                 temp=0.0):
        
        self.CHATGPT_API_KEY = key
        if not self.CHATGPT_API_KEY:
            raise ValueError("CHATGPT_API_KEY is not set")
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp

    def call_chat_gpt_legacy(self, prompt) -> str:
        """used for pre gpt5 models"""
        if not prompt or len(prompt) < 10:
            raise ValueError()

        client = OpenAI(api_key=self.CHATGPT_API_KEY)
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://bitrecs.ai",
                "X-Title": "bitrecs"
            }, 
            model=self.model,
            messages=[
            {
                "role": "user",
                "content": prompt,
            }],
            temperature=self.temp,
            max_tokens=2048
        )

        thing = completion.choices[0].message.content                
        return thing
    

    def call_chat_gpt(self, prompt) -> str:
        if not prompt or len(prompt) < 10:
            raise ValueError()
        
        if "gpt-5" not in self.model.lower():
            return self.call_chat_gpt_legacy(prompt)

        client = OpenAI(api_key=self.CHATGPT_API_KEY)        
        chat_response : Response = client.responses.create(
            extra_headers={
                "HTTP-Referer": "https://bitrecs.ai",
                "X-Title": "bitrecs"
            },
            model=self.model,
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"},
            instructions=self.system_prompt,
            input=prompt,
            #temperature=self.temp, #temp not supported in gpt5
            max_output_tokens=2048
        )
        thing = chat_response.output_text
        return thing