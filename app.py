import sys
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import pickle
import pytesseract
import re
import cv2
from prettytable import PrettyTable

# Model architecture
class DocumentClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DocumentClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Preprocess the image , need to pass to the model
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    input_image = transform(image)
    input_image = input_image.unsqueeze(0)  # Add batch dimension
    return input_image

# Prediction with model
def predict_image(input_image):
    with torch.no_grad():
        output = model(input_image)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()

# check for the receipt or other documents
def interpret_prediction(prediction):
    class_names = ['other_documents', 'receipt']
    return class_names[prediction]


# Predict whether it is receipt or not
def check_receipt_or_not(image):
    image_path = image
    input_image = preprocess_image(image_path).to(device).float()
    prediction = predict_image(input_image)
    result = interpret_prediction(prediction)
    # print(f'The input image is predicted as: {result}')
    if result == "receipt":
        return True
    else:
        return False

def Text_Extractio(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply OCR to the receipt image by assuming column data, ensuring
    # the text is *concatenated across the row* (additionally, for your
    # own images you may need to apply additional processing to cleanup
    # the image, including resizing, thresholding, etc.)
    options = "--psm 4"
    text = pytesseract.image_to_string(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), config=options)
    return text


def extract_vendor_name(text):
    # Split the text into lines
    lines = text

    # Find the first line that starts with a capital letter
    vendor_line = next((line.strip() for line in lines if line and line[0].isupper()), None)

    return vendor_line


def extract_date(text):
    text = str(text)
    date_pattern = re.compile(r'(\b\d{1,2}/\d{1,2}/\d{2,4}\b)|(\d{2}\.\d{2}\.\s*\d{4})')
    match = date_pattern.search(text)
    return match.group() if match else None


def extract_amount(text):
    text = str(text)
    # Regular expression to match lines containing keywords like "Total"
    total_line_pattern = re.compile(r'\bTotal\b.*?([$€£¥]?\s*([+-]?\s*\d+(?:,\d{3})*(?:\.\d{1,2})?))',
                                    re.IGNORECASE | re.DOTALL)

    matches = total_line_pattern.findall(text)

    if matches:
        # Extract the matched amounts
        amounts = [match[1] for match in matches]

        # Convert matched amounts to float for comparison
        amounts_as_float = [float(amount.replace(',', '')) for amount in amounts]
        max_amount = max(amounts_as_float)

        # Find the original formatted string for the maximum amount
        max_amount_str = next(match[0] for match in matches if float(match[1].replace(',', '')) == max_amount)

        return max_amount_str

    return None

# function that fetch the info from receipt
def extract_items(lines):
    item_amount_pattern = re.compile(r'(?P<item>[^\d]+)\s*(?P<amount>[$€£¥]?\s*[+-]?\s*\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                                     re.IGNORECASE)

    items_and_amounts = []

    for line in lines:
        match = item_amount_pattern.search(line)
        if match:
            item_name = match.group('item').strip()
            amount = match.group('amount').replace(',', '')
            items_and_amounts.append({'item_name': item_name, 'amount': float(amount)})

    return items_and_amounts


if __name__ == "__main__":
    with open('./Trained_Model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = "./2_Test.jpeg"

    if check_receipt_or_not(image):
        print("Successfully Found Receipt")
        text = Text_Extractio(image)
        extracted_text = text.splitlines()
        # print(extracted_text)
        vendor_name = extract_vendor_name(extracted_text)
        date = extract_date(extracted_text)
        items = extract_items(extracted_text)
        amount = extract_amount(extracted_text)

        # Organize the extracted information
        extracted_data = {
            'Vendor_name': vendor_name,
            'Date': date,
            'Items': items,
            'Total Amount': amount
        }


        table = PrettyTable()

        # Add columns
        table.field_names = ["Field", "Value"]

        # Add data rows
        for data, value in extracted_data.items():
            if data == 'Items':
                for item in value:
                    table.add_row([data, f"{item['item_name']}: {item['amount']}"])
            else:
                table.add_row([data, value])

        # Print the table
        print(table)

    else:
        print("Upload Image Again, Make sure that is Receipt.")
        sys.exit()


