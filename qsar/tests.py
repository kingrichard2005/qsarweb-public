from django.test import TestCase

# Create your tests here.
class IndexViewsTestCase(TestCase):
    def test_index(self):
        resp = self.client.get('/qsar/')
        self.assertEqual(resp.status_code, 200)