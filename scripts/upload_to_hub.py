from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import HfApi

def upload_model_to_hub():
    # Initialize huggingface api
    api = HfApi()
    
    # Load your fine-tuned model
    model = CLIPModel.from_pretrained("./checkpoints/")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # Push to hub
    model.push_to_hub("[your-username]/clip-breakdance-classifier")
    processor.push_to_hub("[your-username]/clip-breakdance-classifier")

if __name__ == "__main__":
    upload_model_to_hub() 