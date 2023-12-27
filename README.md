# LangChain Business Idea Generator

LangChain is a versatile tool for generating business ideas, conducting analyses, and retrieving financial data for a given country. This script utilizes various modules, including OpenAI's language model, to create a comprehensive workflow. Follow the steps below to set up and use LangChain for your business ideation needs.

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/emon5122/business_idea_generator.git
    cd business_idea_generator
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file and add your API keys for necessary services:

    ```dotenv
    OPENAI_API_KEY=your_openai_api_key
    SERPAPI_API_KEY=your_serpapi_api_key
    ```

## Usage

Run the script by executing:

```bash
python main.py
```

Enter the name of the country when prompted. LangChain will then generate a business idea, perform an analysis, and fetch financial data for the specified country.

## Customization

- **Adjusting Temperature**: You can fine-tune the language model's behavior by changing the temperature parameter in the `OpenAI` instantiation.

- **Modifying Prompts**: Modify the `PromptTemplate` instances in the script to customize the prompts used for business idea generation, analysis, and financial data retrieval.

- **Adding Tools**: Expand the functionality by adding more tools to the `load_tools` function, providing additional data sources for the language model.

## Example

```bash
Enter country name: United States
{
  "business_idea": "E-commerce",
  "business_analysis": "E-commerce is thriving in the United States with a robust online market...",
  "financial_data": "Financial data for E-commerce in the United States until 2023..."
}
```

Feel free to explore and enhance the capabilities of LangChain according to your specific requirements. Happy ideating!

## License

This project is licensed under the [MIT License](LICENSE).
