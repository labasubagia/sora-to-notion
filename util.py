def msg_prefix_progress(processed, total):
    digit = str(total)
    percent = (processed / total) * 100
    processed_str = str(processed).zfill(len(digit))
    return f"{processed_str}/{total} | {percent:06.2f}%"
