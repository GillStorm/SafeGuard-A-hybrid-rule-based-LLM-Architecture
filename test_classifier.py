from core.classifier import classify

print("=" * 60)
print("CLASSIFIER TEST")
print("=" * 60)

# Critical query
result1 = classify("I have chest pain")
print(f"\nQuery:    {result1['query']}")
print(f"Label:    {result1['label']}")
print(f"Severity: {result1['severity']}")
print(f"Domain:   {result1['domain']}")
print(f"Action:   {result1['action']}")

print("-" * 60)

# Ambiguous query
result2 = classify("I feel tired")
print(f"\nQuery:    {result2['query']}")
print(f"Label:    {result2['label']}")
print(f"Severity: {result2['severity']}")
print(f"Domain:   {result2['domain']}")
print(f"Action:   {result2['action']}")

print("-" * 60)

# Off-Topic query
result3 = classify("how to install python")
print(f"\nQuery:    {result3['query']}")
print(f"Label:    {result3['label']}")
print(f"Severity: {result3['severity']}")
print(f"Domain:   {result3['domain']}")
print(f"Action:   {result3['action']}")

print("=" * 60)
print("All tests completed!")
