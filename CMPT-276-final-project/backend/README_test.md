# Factify Test Script

This simple test script allows you to submit a claim to the Factify API and view the verification results. It's helpful for testing your application after making changes like removing the social context feature.

## Prerequisites

Before running this script, make sure:

1. Your Factify Flask API is running (typically on http://localhost:8080)
2. You have the required Python packages installed:

```bash
pip install requests rich
```

## Usage

Run the script with the default claim:

```bash
python test_claim.py
```

Or specify your own claim:

```bash
python test_claim.py --claim "Regular exercise can reduce the risk of heart disease"
```

If your API is running on a different URL:

```bash
python test_claim.py --url "http://custom-url:port" --claim "Your claim here"
```

## Output

The script will display:
- Processing time
- Overall verification verdict and confidence
- Details for each sub-claim, including stance, confidence score, and key points
- Option to view the full JSON response

## Example Claims to Test

Here are some example claims you can try:

- "Coffee consumption reduces the risk of diabetes"
- "Vaccines cause autism in children"
- "Regular exercise can reduce the risk of heart disease"
- "5G networks are harmful to human health"
- "Climate change is primarily caused by human activities"

## Notes

- The verification process can take some time, especially for complex claims
- If you're getting timeout errors, you may need to increase the timeout value in the script 