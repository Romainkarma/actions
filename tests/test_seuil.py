# content of test.py

seuil=0.2

def test_seuil():
    values = [0.1, 0.2, 0.1, 0.1]
    average = sum(values) / len(values)
    assert average <= seuil, f"La moyenne {average} dépasse la valeur seuil de 0.2"
