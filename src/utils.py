import logging
from collections import defaultdict
logger = logging.getLogger(__name__)

def process_content_for_printing(context0: str) -> str:
    """Formats the retrieved context for better readability.

    Args:
        context0: The raw retrieved context.

    Returns:
        str: The formatted context.
    """
    try:
        context1 = context0.split("<h1>")
        context1 = [c1 for c1 in context1 if len(c1)>0]
        context1 = [f'**Source #{i+1}**: \n\n#{c1}' for i, c1 in enumerate(context1)]
        context1 = ''.join(context1).strip()
        context1 = context1.replace("</h2>\nCHUNK:", "\n\nCHUNK:\n\n").replace('</h1>\n<h2>','\n\n## ')
        logger.info("Content formatting successful")
        return(context1)
    except Exception as e:
        logger.error(f"Error formatting content ({context0}) for printing, Error: {str(e)}")
        return "Error in formatting content."

def ld2dl(list0: list[dict])->dict[list]:
    """
    https://g.co/gemini/share/8effebebeca7
    """
    dict_of_lists = defaultdict(list)
    for d in list0:
        for key, value in d.items():
            dict_of_lists[key].append(value)
    return dict_of_lists

# if __name__ == '__main__':
    # logger = logging.getLogger(__name__)
