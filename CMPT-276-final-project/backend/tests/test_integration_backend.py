# tests/test_integration_backend.py

import unittest
from app import app

class TestFlaskAPIIntegration(unittest.TestCase):
    def setUp(self):
        # Create a test client using the Flask application configured for testing.
        self.client = app.test_client()
        app.testing = True

    def test_health_endpoint(self):
        # Verify the /health endpoint returns a status of "ok"
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ok")

    def test_claim_verification_without_claim(self):
        # POST to /api/verification/claim without a "claim" key should return an error (HTTP 400)
        response = self.client.post('/api/verification/claim', json={})
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
