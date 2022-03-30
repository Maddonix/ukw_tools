def frame_number_to_int_str(i:int, zerofill:int = 7) -> str:
    """
    Convert frame number to int string.
    """
    return str(i).zfill(zerofill)