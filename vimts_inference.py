# VimTS Inference and Testing Script
# Test trained model on new images and visualize outputs

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import json
from pathlib import Path

# Import your VimTS components
from backbone import VimTSFeatureExtraction
from loss import VimTSLoss

# Choose your model version (update based on which modules you have trained)
# Option 1: Basic model (Module 1 + 7)
class MinimalVimTSModel(nn.Module):
    """Minimal VimTS model for testing Modules 1 & 7"""
    def __init__(self, num_classes=2, vocab_size=100, max_text_len=25, num_queries=100):
        super().__init__()
        
        # Module 1: Feature Extraction
        self.feature_extractor = VimTSFeatureExtraction(pretrained=True)
        
        # Minimal query generation for testing
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, 256)
        
        # Prediction heads
        self.class_head = nn.Linear(256, num_classes + 1)  # +1 for background
        self.bbox_head = nn.Linear(256, 4)
        self.polygon_head = nn.Linear(256, 16)  # 8 points * 2 coords
        self.text_head = nn.Linear(256, max_text_len * vocab_size)
        
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        
    def forward(self, images):
        batch_size = images.shape[0]
        
        # Module 1: Feature extraction
        features = self.feature_extractor(images)  # [B, 256, H', W']
        
        # Simple query processing
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Global average pooling of features
        pooled_features = features.mean(dim=[2, 3])  # [B, 256]
        
        # Add pooled features to queries
        enhanced_queries = queries + pooled_features.unsqueeze(1)
        
        # Prediction heads
        pred_logits = self.class_head(enhanced_queries)
        pred_boxes = self.bbox_head(enhanced_queries).sigmoid()
        pred_polygons = self.polygon_head(enhanced_queries).sigmoid()
        
        # Text predictions
        text_logits = self.text_head(enhanced_queries)
        pred_texts = text_logits.view(batch_size, self.num_queries, self.max_text_len, self.vocab_size)
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'pred_polygons': pred_polygons,
            'pred_texts': pred_texts
        }

# Option 2: Full model with all modules (if you have trained it)
"""
# Uncomment this if you want to test the full VimTS model
from test_FINAL_Module5 import MinimalVimTSModelWithTaskAdapter

class FullVimTSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MinimalVimTSModelWithTaskAdapter()
    
    def forward(self, images):
        return self.model(images, domain_id=0, task_id=0)
"""

class VimTSInference:
    """VimTS Inference Engine for testing trained models"""
    
    def __init__(self, model_path=None, model_type='minimal', device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        print(f"🖥️ Using device: {self.device}")
        
        # Initialize model
        if model_type == 'minimal':
            self.model = MinimalVimTSModel().to(self.device)
        # elif model_type == 'full':
        #     self.model = FullVimTSModel().to(self.device)
        else:
            raise ValueError("model_type must be 'minimal' or 'full'")
        
        # Load trained weights if provided
        if model_path and os.path.exists(model_path):
            print(f" Loading trained model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f" Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f" Loaded model weights")
        else:
            print(" No model path provided or file not found. Using random weights.")
        
        self.model.eval()
        
        # Confidence thresholds
        self.confidence_threshold = 0.5
        self.text_confidence_threshold = 0.3
    
    def preprocess_image(self, image_path):
        """Preprocess single image for inference"""
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path  # Already PIL Image
    
        # Convert to tensor and ensure values are float32
        image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
        image_tensor = torch.tensor(image_array).permute(2, 0, 1)  # Change order to [C, H, W]
    
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor, image
    
    def postprocess_predictions(self, predictions, image_size):
        """Convert model predictions to interpretable format"""
        img_w, img_h = image_size
        max_size = max(img_h, img_w)
        
        # Get predictions
        pred_logits = predictions['pred_logits'][0]  # Remove batch dim
        pred_boxes = predictions['pred_boxes'][0] 
        pred_polygons = predictions['pred_polygons'][0]
        pred_texts = predictions['pred_texts'][0]
        
        # Apply softmax to get class probabilities
        class_probs = torch.softmax(pred_logits, dim=-1)
        
        # Get text class probabilities (not background)
        text_scores = class_probs[:, 1]  # Assuming class 1 is text
        
        # Filter by confidence
        confident_indices = text_scores > self.confidence_threshold
        
        results = []
        for i in range(len(pred_logits)):
            if confident_indices[i]:
                # Scale coordinates to image size
                box = pred_boxes[i] * max_size
                polygon = pred_polygons[i] * max_size
                
                # Convert text predictions to string
                text_logits = pred_texts[i]  # [max_text_len, vocab_size]
                text_chars = torch.argmax(text_logits, dim=-1)  # [max_text_len]
                
                # Convert to string (simple approach)
                text = ''.join([chr(min(max(char.item(), 32), 126)) for char in text_chars if char.item() > 0])
                text = text.strip()
                
                result = {
                    'confidence': text_scores[i].item(),
                    'bbox': box.cpu().numpy(),
                    'polygon': polygon.cpu().numpy().reshape(-1, 2),
                    'text': text,
                    'class_probs': class_probs[i].cpu().numpy()
                }
                results.append(result)
        
        return results
    
    def predict_single_image(self, image_path):
        """Run inference on single image"""
        print(f"🔍 Processing: {image_path}")
        
        # Preprocess
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Postprocess
        results = self.postprocess_predictions(predictions, original_image.size)
        
        print(f" Found {len(results)} text detections")
        
        return results, original_image
    
    def visualize_results(self, image, results, save_path=None, show_plot=True):
        """Visualize detection results on image"""
        # Create a copy for drawing
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Try to load a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw detections
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, result in enumerate(results):
            color = colors[i % len(colors)]
            confidence = result['confidence']
            bbox = result['bbox']
            text = result['text']
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw text label
            label = f"{text} ({confidence:.2f})"
            
            # Get text size for background
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
            
            # Draw background for text
            draw.rectangle([x1, y1-text_h-5, x1+text_w+5, y1], fill=color)
            
            # Draw text
            draw.text((x1+2, y1-text_h-3), label, fill='white', font=font)
        
        # Show or save
        if show_plot:
            plt.figure(figsize=(12, 8))
            plt.imshow(vis_image)
            plt.axis('off')
            plt.title(f'VimTS Detection Results ({len(results)} detections)')
            plt.tight_layout()
            plt.show()
        
        if save_path:
            vis_image.save(save_path)
            print(f" Saved visualization to: {save_path}")
        
        return vis_image
    
    def test_multiple_images(self, image_folder, output_folder=None, limit=None):
        """Test model on multiple images"""
        image_folder = Path(image_folder)
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_folder.glob(f'*{ext}')))
            image_files.extend(list(image_folder.glob(f'*{ext.upper()}')))
        
        if limit:
            image_files = image_files[:limit]
        
        print(f" Found {len(image_files)} images to test")
        
        all_results = []
        
        for i, image_path in enumerate(image_files):
            print(f"\n--- Image {i+1}/{len(image_files)} ---")
            
            try:
                # Run inference
                results, original_image = self.predict_single_image(image_path)
                
                # Visualize and save
                if output_folder:
                    output_path = os.path.join(output_folder, f"result_{image_path.stem}.png")
                    self.visualize_results(original_image, results, save_path=output_path, show_plot=False)
                else:
                    self.visualize_results(original_image, results, show_plot=True)
                
                # Store results
                all_results.append({
                    'image_path': str(image_path),
                    'detections': len(results),
                    'results': results
                })
                
                # Print summary
                if results:
                    print(" Detected texts:")
                    for j, result in enumerate(results):
                        print(f"   {j+1}. '{result['text']}' (conf: {result['confidence']:.3f})")
                else:
                    print(" No text detected")
                
            except Exception as e:
                print(f" Error processing {image_path}: {str(e)}")
                continue
        
        # Summary
        total_detections = sum(r['detections'] for r in all_results)
        avg_detections = total_detections / len(all_results) if all_results else 0
        
        print(f"\n SUMMARY:")
        print(f"   Images processed: {len(all_results)}")
        print(f"   Total detections: {total_detections}")
        print(f"   Average detections per image: {avg_detections:.1f}")
        
        return all_results

def main():
    """Main testing function"""
    print(" VimTS Model Testing")
    print("=" * 50)
    
    # Configuration
    model_path = None  # Path to your trained model checkpoint (or None for random weights)
    test_image_path = "/content/sample/test/000011.jpg"  # Single image to test
    test_folder_path = "/content/sample/test"  # Folder with test images
    output_folder = "/content/inference_results/"  # Where to save visualization results
    
    # Initialize inference engine
    inference = VimTSInference(
        model_path=model_path,
        model_type='minimal',  # 'minimal' or 'full'
        device='auto'
    )
    
    # Test options
    print("Choose testing option:")
    print("1. Test single image")
    print("2. Test multiple images from folder")
    print("3. Test with sample dummy image")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        # Test single image
        if os.path.exists(test_image_path):
            results, original_image = inference.predict_single_image(test_image_path)
            inference.visualize_results(original_image, results)
        else:
            print(f" Image not found: {test_image_path}")
    
    elif choice == "2":
        # Test multiple images
        if os.path.exists(test_folder_path):
            inference.test_multiple_images(test_folder_path, output_folder, limit=5)
        else:
            print(f" Folder not found: {test_folder_path}")
    
    elif choice == "3":
        # Test with dummy image
        print(" Creating dummy test image with text...")
        
        # Create a simple test image with text
        dummy_image = Image.new('RGB', (640, 480), color='white')
        draw = ImageDraw.Draw(dummy_image)
        
        # Add some text to detect
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
        
        draw.text((50, 100), "HELLO WORLD", fill='black', font=font)
        draw.text((50, 200), "TEST TEXT", fill='blue', font=font)
        draw.text((50, 300), "VimTS Demo", fill='red', font=font)
        
        # Test on dummy image
        results, original_image = inference.predict_single_image(dummy_image)
        inference.visualize_results(original_image, results)
    
    else:
        print(" Invalid choice")

if __name__ == "__main__":
    main()
