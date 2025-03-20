import unittest
from app import openalex_service

class TestOpenAlexServiceUnit(unittest.TestCase):
    def test_reconstruct_abstract_from_inverted_index(self):
        # Given an inverted index where the word "Hello" appears at positions 0 and 2,
        # and "World" appears at position 1, the reconstructed abstract should be "Hello World Hello".
        inverted_index = {
            "Hello": [0, 2],
            "World": [1]
        }
        expected = "Hello World Hello"
        result = openalex_service._reconstruct_abstract_from_inverted_index(inverted_index)
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
