### Can small language models detect mistakes in their Chain-of-Thought?
- Can small models detect mistakes in their CoT? 
- Can you isolate circuits that look for mistakes in its reasoning? 
- If we patch CoT and corrupt it with wrong values, can the model detect these corrupted wrong values while reasoning?
- How do models internally detect and correct mistakes?
- Say in toy problem of addition, at what stage in the forward pass does error detection occur?
- Which layers are responsible for it? How does the model recognize and correct mistakes?
- How do larger models get better at this?
- How do distilled models / math instructed models do this?
- What makes distilled reasoning models so much better at showing this behaviour?
- Over larger models, can you observe any intersting simialar behaviour?
- Can you train probes to detect when a language model's output is not reflective of its internal state? Or when the outputs are misaligned?

