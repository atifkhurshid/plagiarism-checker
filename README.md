# Plagiarism Checker
This is a Python-based plagiarism checker that uses document similarity to identify potential instances of plagiarism. The tool uses cosine similarity to evaluate the similarity between pairs of files in a specified directory. The similarity scores range from 0 to 1, where 1 indicates identical content and 0 indicates no similarity.

## Installation
Clone the repository and install required packages using
```bash
pip install -r requirements.txt
```

## PDF File Matcher
This program takes a directory of PDF files and compares each file against every other file to find potential matches. Comparisons are made based on cosine similarity of TF-IDF feature vectors.

### Usage
Run the program from the command line with the following arguments:

- --dir: The directory containing the PDF files to be checked.
- --threshold: A similarity threshold between 0 and 1. Only file pairs with a similarity score above this threshold will be reported.

### Example
```bash
python pdf_matching.py --dir /path/to/pdf/files/ --threshold 0.8
```

This will scan the /path/to/pdf/files directory, comparing all PDF files within it. Only file pairs with a similarity score of 0.8 or higher will be shown.

### Output
The program will return a list of file pairs that match the specified threshold, along with their cosine similarity scores.

```yaml
File1.pdf and File2.pdf: Similarity = 0.85
File3.pdf and File5.pdf: Similarity = 0.92
```

## License
This project is licensed under the MIT License. Feel free to contribute to the project by submitting issues or pull requests.