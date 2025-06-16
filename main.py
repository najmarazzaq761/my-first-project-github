from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from agents import (
    stack_detector, test_writer, build_pipeline, deploy_agent, yaml_writer, explain_agent
)

app = FastAPI()

openai.api_key = os.getenv("GROQ_API_KEY")
openai.base_url = "https://api.groq.com/openai/v1/"

class RepoRequest(BaseModel):
    repo_link: str
    platform: str
    files: list[str]

@app.post("/detect-stack")
async def detect_stack(files: list[str]):
    response = openai.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": stack_detector.get_stack_prompt()},
            {"role": "user", "content": f"files: {files}"}
        ]
    )
    return response.choices[0].message.content

@app.post("/generate-tests")
async def generate_tests(files: list[str]):
    response = openai.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": test_writer.get_test_prompt()},
            {"role": "user", "content": f"Generate test YAML for {files}"}
        ]
    )
    return response.choices[0].message.content

@app.post("/build-pipeline")
async def build_pipeline_endpoint():
    response = openai.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": build_pipeline.get_build_prompt()},
            {"role": "user", "content": "Add build steps for Python app with requirements.txt"}
        ]
    )
    return response.choices[0].message.content

@app.post("/deploy-config")
async def deploy_config(platform: str):
    response = openai.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": deploy_agent.get_deploy_prompt(platform)},
            {"role": "user", "content": f"Deploy my app to {platform}"}
        ]
    )
    return response.choices[0].message.content

@app.post("/combine-yaml")
async def combine_yaml(test_steps: str, build_steps: str, deploy_steps: str):
    response = openai.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": yaml_writer.get_yaml_writer_prompt()},
            {"role": "user", "content": f"Test Steps:\n{test_steps}\nBuild Steps:\n{build_steps}\nDeploy Steps:\n{deploy_steps}"}
        ]
    )
    return response.choices[0].message.content

@app.post("/explain")
async def explain_pipeline(yaml_content: str):
    response = openai.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": explain_agent.get_explanation_prompt()},
            {"role": "user", "content": yaml_content}
        ]
    )
    return response.choices[0].message.content
