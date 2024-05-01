# Algorithmic Pipeline for Receipt Information Extraction

## Pre-processing of the Image
 
### Input
- JPG images from the images directory.

### Processing Steps
- **Rotation Correction**: Automatically detect and correct the orientation of the receipt using image rotation detection algorithms. (images_01,images_04, images_00)
- **Receipt Segmentation**: In case of multiple receipts in a single image, segment each receipt using contour detection or similar methods.
- **Image Quality Enhancement**: Adjust brightness, contrast, and sharpness to enhance text visibility.
- **Noise Reduction**: Apply filters to reduce image noise that can interfere with text recognition.

### Output
- Images prepared for text extraction, with each receipt isolated and properly oriented.

## Text Recognition (OCR)

### Input
- Pre-processed images from the previous step.

### Processing Steps
- **Optical Character Recognition (OCR)**: Apply OCR technology (e.g. Surya, Tesseract, an open-source OCR engine) to convert images into text.
- **Text Cleanup**: Implement regular expressions and text cleaning techniques to remove noise. Be sure not to remove ".", it might give false amount. (100.000 wouold become 100000 Yen)
- **Character Segmentation**: Isolate individual characters or words within the identified text regions. 
- **Character Recognition**: Convert segmented characters to digital text using OCR technology.

### Output
- Raw text extracted from each receipt.

## Information Extraction

### Input
- Raw text data from OCR.

### Processing Steps
- **Entity Recognition**: Using NER on japanese text or some other methods(e.g., use regular expressions to locate currency symbols followed by numerical values for amounts) to extract total amount and organization name
- **Data Extraction**: Extract and parse the 'issuer company' name and 'total amount' from the recognized patterns. These are most likely to be present at top of receipt (Organization Name) and bottom of receipt(Bottom of receipt) so we can modify out data extractor to focus more on these parts.
    This can also be done by LLM which will likely to perform better than other techniques but performance wise would be slower and costly.
- **Keyword Identification**: Create a list of keywords or phrases that typically indicate the issuer company's name.

### Output
- Structured data containing the issuer company and total amounts for each receipt.

## Post-processing and Output

### Input
- Extracted data from the information extraction step.

### Processing Steps
- **Validation and Correction**: Check for common OCR errors in company names and amounts, applying correction rules as necessary.
    This method can also be done by low cost LLM.
- **Data Normalization**: Standardize data formats (e.g., convert all amounts to a standard currency format).
- **Error Correction**: Use error detection algorithms to correct common OCR mistakes.
- **Formatting**: Format the data into a structured array or JSON object.

### Output
- Final structured array where each object holds the "total amount" and "issuer company" values for each receipt.

## Storage

- **Action**: Store the output data in a specified format for further use or integration with other systems.

## Quality Assurance

- **Manual Verification**: Include an optional step where a human operator can verify and correct OCR results.
- **Feedback Loop**: Use incorrect extractions as feedback to improve the OCR model.
