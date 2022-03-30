def split_size_str(size_str):
    if size_str == "unknown":
        result = None
    elif size_str == "<5":
        result = (0, 4.99)
    elif size_str == "5-10":
        result = (5, 10)
    elif size_str == ">10-20":
        result = (10.01, 20)
    elif size_str == ">20":
        result = (20.01, 999)

    return result
