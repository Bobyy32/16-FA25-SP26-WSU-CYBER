<|user_query|>
Here's my Python code using imgaug for image augmentation. I'd like to refactor this to be more maintainable and add proper error handling.

<|model_response|>
I can help you refactor this code. Let me suggest several improvements:
1. Remove unnecessary `from __future__` imports (they're legacy Python 2/3 compatibility imports)
2. Add proper docstrings and type hints
3. Use context managers for better resource management
4. Add input validation
5. Simplify the augmentation parameters handling

<|user_query|>
What about the numpy array resizing and image shape handling? Should we keep the imgaug functionality?

<|model_response|>
Great question. The imgaug library is quite powerful, but we can simplify the implementation while keeping core functionality. Let me show you the refactored version with better structure and maintained functionality:

<|user_query|>
Perfect. Can you show me the complete refactored version?

<|model_response|>
Certainly! Here's the complete refactored code with all improvements applied: