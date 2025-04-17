import os
import torch
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
import gradio as gr

class VegetableClassifier:
    def __init__(self, num_classes=15, model_name='resnet50'):
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, num_classes)
        elif model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=False)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model = self.model.to(self.device)
        
        # Define transformation for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Class names (replace with your actual class names)
        self.class_names = [
            'bean', 'bitter_gourd', 'bottle_gourd', 'brinjal', 'broccoli', 
            'cabbage', 'capsicum', 'carrot', 'cauliflower', 'cucumber', 
            'papaya', 'potato', 'pumpkin', 'radish', 'tomato'
        ]
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def predict(self, image_path):
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            # If image_path is already a PIL Image
            img = image_path.convert('RGB')
        
        # Apply transformations
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get top predictions
        top_p, top_class = probs.topk(3)
        
        # Convert to lists
        top_p = top_p.cpu().numpy()
        top_class = top_class.cpu().numpy()
        
        # Get class names and confidence scores
        predicted_class = self.class_names[top_class[0]]
        confidence = top_p[0]
        
        # Get top 3 predictions
        top_3 = [(self.class_names[top_class[i]], float(top_p[i])) for i in range(3)]
        
        return top_3, img

# Function to process image and return predictions
def predict_vegetable(image):
    model_path = "best_vegetable_model.pth"  # Adjust path as needed
    
    # Create and load model
    classifier = VegetableClassifier(num_classes=15, model_name='resnet50')
    
    try:
        classifier.load_model(model_path)
    except Exception as e:
        return None, f"Error loading model: {str(e)}"
    
    # Make prediction
    predictions, img = classifier.predict(image)
    
    # Create a formatted result string
    result = ""
    for i, (veg, conf) in enumerate(predictions):
        result += f"{i+1}. {veg.capitalize()}: {conf*100:.2f}%\n"
    
    # Create visualization of predictions
    fig = plt.figure(figsize=(10, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Prediction: {predictions[0][0].capitalize()}")
    plt.axis('off')
    
    # Display bar chart
    plt.subplot(1, 2, 2)
    classes = [pred[0].capitalize() for pred in predictions]
    scores = [pred[1] for pred in predictions]
    
    plt.barh(classes, scores, color='skyblue')
    plt.xlim(0, 1.0)
    plt.xlabel('Confidence')
    plt.title('Top 3 Predictions')
    
    # Add percentage labels to bars
    for i, score in enumerate(scores):
        plt.text(score + 0.01, i, f'{score*100:.1f}%')
    
    plt.tight_layout()
    
    return fig, result

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_vegetable,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Plot(label="Prediction Result"),
        gr.Textbox(label="Top 3 Predictions")
    ],
    title="🥦 Vegetable Classifier",
    description="Upload an image of a vegetable to identify it using our trained model.",
    examples=[
        ["dataset/validation/Carrot/1201.jpg"],
        ["dataset/validation/Tomato/1201.jpg"],
        ["dataset/validation/Potato/1026.jpg"]
    ],
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    
    # Launch the interface
    iface.launch(share=False)