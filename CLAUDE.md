# GRPODx Project Guidelines

## Running Commands
- Run main training: `python medical_grpo.py --openai_api_key YOUR_API_KEY`
- Run trained model: `python doctor_chat.py --model_path doctor_lora_final`
- Set model temperature: `python doctor_chat.py --temperature 0.7`
- List saved checkpoints: `ls doctor_outputs`

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then local modules
- **Typing**: Use Python type hints for all function parameters and returns
- **Naming**: 
  - snake_case for variables and functions
  - CamelCase for classes
  - UPPERCASE for constants
- **Documentation**: Use docstrings with parameter descriptions and return types
- **Structure**: Use section comments with `#####` separators for code organization
- **Error Handling**: Use try-except blocks with specific exceptions
- **Logging**: Use the logging module with appropriate levels (INFO, WARNING, ERROR)
- **Formatting**: Follow PEP 8 guidelines with 4-space indentation