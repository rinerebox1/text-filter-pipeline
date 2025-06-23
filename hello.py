def hello(name):
    """
    Greets a person by name.
    
    Args:
        name (str): The name of the person to greet
        
    Returns:
        str: A greeting message
    """
    # Handle empty string case
    if not name:
        return "Hello, Guest!"
    
    # Format the name with proper capitalization
    first_char = name[0].upper()
    rest_chars = name[1:].lower()
    formatted_name = first_char + rest_chars
    
    return f"Hello, {formatted_name}!"
