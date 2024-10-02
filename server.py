import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set random seed for reproducibility
torch.random.manual_seed(0)

# Load your trained model and tokenizer
model_name = "DanielShaw98/phi-3.5-law"  # Update with your model's name
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Use 'auto' to leverage GPU if available
    torch_dtype="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the conversation messages
# messages = [
#     {"role": "system", "content": "You are Legal AI. Your job is to help lawyers by identifying specific clauses in merger and acquisition contracts. Please identify the desired clauses and also provide an explanation for this choice based on the prompt."},
#     {"role": "user", "content": "Review the provided text and identify all clauses related to termination rights and conditions. Return the exact start and end line numbers of the relevant clause within the chunk. If you cannot find anything relevant, please respond with 'nothing found.' Please provide details in the following JSON format: {  \"relevant_chunks_found\": <number>,  \"entries\": [    {      \"page\": <page_number>,      \"line_start\": <clause_start_line_within_chunk>,      \"line_end\": <clause_end_line_within_chunk>,      \"clause\": <clause_text>,      \"explanation\": <explanation_text>    }  ]}\n\n Chunk: Transaction but all the conditions therein have been satisfied or complied with, \nor confirmed no such clearance is required in accordance with the applicable \ncompetition legislation, or has not objected to the Transaction within the time \nperiod prescribed by law. \n227876-4-1460-v9.0 \n- 30 - \n70-40688062 \n \nFor the purposes of clauses 4.1.10 to 4.1.12 (inclusive) only, \"Transaction\" shall \nbe limited to the part or parts of the Transaction required to be notified to the \nCommission, COFECE or the competent competition authority of Vietnam (as \nappropriate). \nNo material breach \n4.1.13 no Purchaser Covenant Breach and no Purchaser Material Breach having \noccurred; and \n4.1.14 no Chrysaor Covenant Breach and no Chrysaor Material Breach having \noccurred. \n4.2 \nAny Regulatory Condition or Antitrust Condition may be waived at any time on or \nbefore 17.00 on the Longstop Date by written agreement of the Company and the \nPurchaser. Any Chrysaor Material Breach may be waived at any time on or before \n17.00 on the Longstop Date by the Purchaser by notice in writing to the Company.  Any \nPurchaser Material Breach may be waived at any time on or before 17.00 on the \nLongstop Date by the Company by notice in writing to the Purchaser. \n4.3 \nIf, at any time, any party becomes aware of a fact, matter or circumstance that could \nreasonably be expected to prevent or delay the satisfaction of a Condition, it shall \ninform the others of the fact, matter or circumstance as soon as reasonably practicable. \n4.4 \nIf a Condition has not been satisfied or (if capable of waiver) waived by 17.00 on the \nLongstop Date or becomes impossible to satisfy before that time, either the \nHarbour/Chrysaor Parties or the Purchaser may terminate this Agreement by notice in \nwriting to that effect to the other, save that the Harbour/Chrysaor Parties may only \nterminate this Agreement: (i) on the basis of the Whitewash Condition not having been \nsatisfied by 17.00 on the Longstop Date or having become impossible to satisfy before \nthat time; and (ii) on the basis of the Circular Condition and/or the FCA Admission \nCondition not having been satisfied by 17.00 on the Longstop Date or having become \nimpossible to satisfy before that time, in each case, only if the Harbour/Chrysaor Parties \nhave complied with the relevant provisions of clause 5 and/or the Purchaser has not \ncomplied with the relevant provisions of clause 5. \n4.5\n\n Chunk Meta-Data:\nPage Start: 30\n Page End: 31\nLine Start: 1405\n Line End: 1445"},
# ]

# Create a pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Set generation arguments
generation_args = {
    "max_new_tokens": 200,
    "return_full_text": False,
    # "temperature": 0.0,
    "do_sample": False,
}

# Format the messages for the model input
# formatted_input = f"{messages[0]['role']}: {messages[0]['content']}\n" \
#                   f"{messages[1]['role']}: {messages[1]['content']}"
# formatted_input = (
#     f"<|system|>\n{messages[0]['content']}<|end|>\n"
#     f"<|user|>\n{messages[1]['content']}<|end|>"
# )
# print("Formatted Input:", formatted_input)


# Generate output
# output = pipe(formatted_input, **generation_args)
# output = pipe(messages, **generation_args)

# Print the generated text
# print(output[0])

simple_prompt = "What is the purpose of a termination clause in contracts?"
output = pipe(simple_prompt, **generation_args)
print(output[0])
