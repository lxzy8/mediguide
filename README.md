base_model: tiiuae/falcon-rw-1b library_name: peft
Model Card for MediGuide E-Doctor
This model card provides details about the fine-tuned Large Language Model developed for the MediGuide E-Doctor project, aimed at providing preliminary, text-based medical guidance.

Model Details
Model Description
The MediGuide E-Doctor model is a fine-tuned version of the tiiuae/falcon-rw-1b (1 Billion parameters) Large Language Model. It has been adapted to understand and respond to medical inquiries in a conversational format, focusing on providing accurate, contextually appropriate, and professionally worded advice while emphasizing that it does not replace professional diagnosis.

Developed by: Khushal

Model type: Causal Language Model (decoder-only Transformer)

Language(s) (NLP): English

License: The base model tiiuae/falcon-rw-1b is typically under the Apache 2.0 license. The fine-tuned adapters inherit this.

Finetuned from model: tiiuae/falcon-rw-1b

Uses
Direct Use
The MediGuide E-Doctor is intended for direct use as a preliminary conversational AI tool. Users can input free-text medical questions or symptom descriptions, and the model will generate informative, contextually relevant, and professionally worded responses. It serves as an initial guidance system, not a diagnostic tool.

Downstream Use
The fine-tuned LoRA adapters can be integrated into larger healthcare platforms or applications requiring conversational AI capabilities for patient triage, FAQ answering, or preliminary information dissemination in a medical context.

Out-of-Scope Use
This model is not intended for:

Providing medical diagnoses, treatment plans, or emergency medical advice.

Handling sensitive Protected Health Information (PHI) unless explicitly designed and secured within a HIPAA-compliant environment.

Replacing qualified healthcare professionals.

Use in life-critical systems where errors could lead to direct harm.

Generating content outside of the medical guidance domain.

Bias, Risks, and Limitations
Large Language Models, even when fine-tuned, can inherit biases from their pre-training data or inadvertently learn biases present in the fine-tuning data. Specific risks include:

Hallucinations: Generating factually incorrect or nonsensical information, which is highly critical in a medical context.

Overgeneralization/Under-specification: Providing generic advice when specific guidance is needed, or vice-versa.

Lack of Real-time Medical Knowledge: The model's knowledge is static to its last training update; it does not have access to real-time medical developments or specific patient records.

Ethical Concerns: Misinterpretation by users who might mistake preliminary guidance for definitive medical advice.

Recommendations
Users (both direct and downstream) should be made aware of the risks, biases, and limitations of the model.

Clear Disclaimers: Every interaction should prominently feature a disclaimer stating that the AI does not provide diagnoses and is not a substitute for professional medical advice.

Human Oversight: For critical applications, human healthcare professionals should review and validate AI-generated responses.

Continuous Monitoring: Regular evaluation and monitoring of model outputs for accuracy, safety, and adherence to guidelines.

Data Quality: Ongoing efforts to expand and refine the training dataset to reduce biases and improve accuracy.

How to Get Started with the Model
To get started with the fine-tuned MediGuide E-Doctor model, you will need the base tiiuae/falcon-rw-1b model and the saved LoRA adapters.

Install necessary libraries:

!pip install -U transformers peft bitsandbytes datasets accelerate

Load the model and tokenizer:

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

model_name = "tiiuae/falcon-rw-1b"
adapter_path = "./mediguide-falcon" # Or the path where you saved your adapters

peft_config = PeftConfig.from_pretrained(adapter_path)
bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval() # Set model to evaluation mode

Perform inference:

user_question = "What are the common symptoms of a flu?"
formatted_input = f"Patient: {user_question}\nDoctor:" # Ensure format matches training data

inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

Training Details
Training Data
The model was fine-tuned on the conversation_data.csv dataset.
This dataset comprises 9,958 conversational turn pairs, formatted as patient questions and doctor responses. The data was preprocessed to combine these into a single text column for causal language modeling, adopting the format "Patient: [Patient_Answer]\nDoctor: [Doctor_response]". The dataset is designed to teach the model how to respond in a medically relevant and conversational manner.

Training Procedure
The model was fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) specifically with the LoRA (Low-Rank Adaptation) method, combined with 8-bit quantization. This allowed for efficient adaptation of the large base model on limited computational resources.

Preprocessing [optional]
The conversation_data.csv was loaded using pandas.

The Patient_Answer and Doctor_response columns were concatenated into a single text column with a specific conversational prefix "Patient: " and "Doctor: ".

The text was then tokenized using the tiiuae/falcon-rw-1b tokenizer, with padding and truncation applied to max_length=512.

For causal language modeling, the input_ids were duplicated to create the labels for the Trainer.

Training Hyperparameters
Base Model: tiiuae/falcon-rw-1b

PEFT Method: LoRA

Quantization: 8-bit (BitsAndBytesConfig, load_in_8bit=True, llm_int8_threshold=6.0)

LoRA Parameters:

r=8

lora_alpha=32

target_modules=["query_key_value"]

lora_dropout=0.05

bias="none"

task_type=TaskType.CAUSAL_LM

Training Regime: Mixed precision (likely fp16 automatically handled by bitsandbytes and accelerate)

Training Arguments:

output_dir="./mediguide-falcon"

per_device_train_batch_size=4

num_train_epochs=3

logging_steps=10

save_strategy="epoch"

learning_rate=2e-4

report_to="none"

Speeds, Sizes, Times 
Total Trainable Parameters (LoRA Adapters): 1,572,864 (~0.12% of the base model's total 1,313,198,080 parameters)

Training Runtime (initial run on 2.5K data): ~44 minutes (on Google Colab T4 GPU)

Checkpoint Size: The LoRA adapters themselves are relatively small (a few MBs), while the base model (2.62 GB) is loaded separately.

Evaluation
Testing Data, Factors & Metrics
Testing Data
(Note: A dedicated test set was not explicitly defined during the training process. For a robust evaluation, a portion of conversation_data.csv should be held out strictly for testing.)

Factors
Domain Specificity: Performance on medical queries.

Conversational Flow: Ability to maintain a coherent dialogue.

Response Quality: Accuracy, professionalism, and inclusion of disclaimers.

Metrics
ROUGE-1 F1 Score: Measures unigram overlap between generated and reference responses (for content overlap).

ROUGE-L F1 Score: Measures the longest common subsequence (LCS) overlap (for structural similarity).

Perplexity (PPL): An intrinsic measure of the model's fluency and confidence in predicting text; lower is better.

Inference Latency (Avg.): The average time taken to generate a response, crucial for user experience.

Human Evaluation: (Recommended for future work) Subjective assessment by human experts for medical accuracy, safety, and tone.

Results
(Note: The following results are placeholders. Actual values should be obtained through a rigorous evaluation process on a held-out test set.)

Metric / Strategy

Falcon-rw-1b (LoRA, 8-bit Quant.)

Final Training Loss

0.6386 (on previous 2.5K data)

ROUGE-1 F1 Score

[Placeholder]

ROUGE-L F1 Score

[Placeholder]

Perplexity (PPL)

[Placeholder]

Inference Latency (Avg.)

[Placeholder] ms

GPU VRAM Usage (Approx.)

~8-10 GB (during training on Colab T4)

Summary
The fine-tuning process successfully adapted the Falcon-rw-1b model to the medical conversational domain using efficient PEFT techniques. The model showed clear signs of learning from the training data, as indicated by the decreasing training loss. Further evaluation on a dedicated test set is required to quantitatively assess its performance on unseen data across various metrics and ensure it meets the stringent requirements for medical guidance systems.

Environmental Impact
Carbon emissions can be estimated using the Machine Learning Impact calculator presented in Lacoste et al. (2019).

Hardware Type: GPU (e.g., NVIDIA Tesla T4 or P100)

Hours used: [More Information Needed - sum up training hours]

Cloud Provider: Google Cloud (via Google Colab/Kaggle)

Compute Region: [More Information Needed - often unknown/variable for Colab/Kaggle free tier]

Carbon Emitted: [More Information Needed - calculate using the link and above info]

Technical Specifications 
Model Architecture and Objective
The model architecture is a decoder-only Transformer, characteristic of causal language models. Its objective during fine-tuning was to predict the next token in a sequence, thereby learning to complete conversational turns (patient question followed by doctor response) based on the provided training data.

Compute Infrastructure
Hardware
Training: Google Colab / Kaggle Notebooks (NVIDIA Tesla T4 or P100 GPUs)

Development: MacBook Air M1 (for initial setup and understanding, but not for heavy training due to GPU incompatibility with bitsandbytes)

Software
Programming Language: Python 3.x

Machine Learning Frameworks: PyTorch

Transformers & Fine-Tuning Libraries: Hugging Face transformers, Hugging Face peft, bitsandbytes (for 8-bit quantization), accelerate, datasets, pandas.

Development Environment: Google Colab / Kaggle Notebooks

Citation [optional]
BibTeX:

@article{tiiuae2023falcon,
  title={Falcon-40B: an Open-Source Large Language Model},
  author={Almazrouei, Ebtesam and Alobeidli, Hamza and Alshamsi, Abdulaziz and Al-Kaabi, Ruba and Al-Falahi, Ahmed and Al-Hammadi, Raeda and Al-Marzooqi, Khulood and Al-Teneiji, Nada and Al-Hajri, Sana and Al-Nuaimi, Marwan and Al-Mahmoud, Mohammad and Al-Hammadi, Youssef and Al-Dahmani, Amna and Al-Dhaheri, Nada and Al-Zaabi, Feras and Al-Ahbabi, Sara and Al-Ameri, Maryam and Alowais, Ebrahim and Aljuneibi, Salem and Ayoub, Abdulla and Bellare, Pratyush and Bica, Andrei and Chaaraoui, Rawan and Farooq, Moin and Faruqui, Manaal and El-Dardiry, Nayer and Ghazal, Imad and El Gayar, Muhammad and Hamadi, Elias and Hareb, Omar and Ibrahim, Fahad and Jha, Rohit and Kazma, Ghada and Khayyat, Ziad and Krishna, Rakesh and Kumar, Sanket and El Kweifi, Emad and Laanait, Noura and Lakhani, Devam and Li, Jian and Li, Weizhong and Al-Madhi, Abdulrahman and Mansour, Bayan and Mezghani, Hicham and Mourad, Imene and Nahlus, Shady and Naouali, Najia and Nedi, Daniel and Ouzaa, Fares and El Mghazli, Soufiane and Rached, Hadi and Rameh, Rawan and Remli, Nizar and Saeed, Omar and Said, Mohamed and Shah, Hira and Shakya, Saurabh and Sulaiman, Ibrahim and Tarek, Hani and Tazi, Fouad and Toutounji, Ibrahim and Varma, Vishnu and Varma, Vishnu and Yousif, Muhammad and Zhao, Zhiting and Zolfaghari, Reza and Al-Zoubi, Khaled and Zomaya, Albert and Al-Shaikh, Anas},
  journal={arXiv preprint arXiv:2307.03960},
  year={2023}
}

APA:

TII UAE. (2023). Falcon-40B: an Open-Source Large Language Model. arXiv preprint arXiv:2307.03960.

Glossary 
LLM (Large Language Model): A type of artificial intelligence model trained on vast amounts of text data to understand, generate, and process human language.

PEFT (Parameter-Efficient Fine-Tuning): A set of techniques that enable efficient adaptation of large pre-trained models to downstream tasks without fine-tuning all of the model's parameters.

LoRA (Low-Rank Adaptation): A specific PEFT method that injects small, trainable matrices into the transformer architecture's layers.

8-bit Quantization: A method to reduce the memory footprint of a model by storing its parameters with reduced precision (8 bits per number).

Causal Language Model: A type of language model that predicts the next token in a sequence based on the preceding tokens.

ROUGE: A set of metrics for evaluating text summarization and generation by comparing generated text to a reference text based on overlapping units (words, n-grams, or sequences).

Perplexity (PPL): A measure of how well a probability distribution predicts a sample. In language modeling, a lower perplexity indicates a better model.

Latency: The time delay between an input and a system's response.

Unified Memory: A type of memory architecture (e.g., in Apple Silicon) where CPU and GPU share the same physical memory pool.

HIPAA: Health Insurance Portability and Accountability Act, a US law that sets standards for sensitive patient data protection. "HIPAA-equivalent" refers to adhering to similar strict data privacy standards.

Framework versions
PEFT 0.14.0

Transformers (latest stable via pip install -U)

BitsAndBytes (version 0.41.3 or latest stable via pip install -U)

PyTorch (compatible with Colab/Kaggle GPU)

Python 3.10+ (as used in Colab/Kaggle)
