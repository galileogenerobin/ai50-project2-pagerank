import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Create a dictionary of all pages in the corpus and initialize the probabilities to (1 - damping_factor) / (count of all web pages in the corpus)
    # This will be our output dictionary
    output = {}
    web_pages_count = len(corpus)
    for web_page in corpus:
        # This is our baseline probability value
        output[web_page] = (1 - damping_factor) / web_pages_count

    # Get the probability of visiting pages via the current page (base probability = damping_factor)
    # We will only do this if there are linked pages from the given page
    if not corpus[page] == set():
        # Number of linked pages
        linked_pages_count = len(corpus[page])
        for linked_page in corpus[page]:
            # The probability of visting the linked page via the current page is the damping factor equally distributed across all linked pages
            # And we will add this probability to the base probability set above
            output[linked_page] = output[linked_page] + (damping_factor / linked_pages_count)
    else:
        # No linked pages; so we normalize the previously set probabilities to sum up to 1
        for web_page in corpus:
            output[web_page] = 1 / web_pages_count

    return output
    # raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Set up our output dictionary
    output = {}
    for web_page in corpus:
        output[web_page] = 0

    # Start sampling
    for i in range(n):
        if i == 0:
            # Selection for first sample will be equally distributed
            # We only want to get the page, not the page:value pair
            sample = random.choice([page for page in corpus])
        else:
            # For succeeding samples, the weight of selection for samples will be based on the transition model
            sample_transition = transition_model(corpus, sample, damping_factor)

            # From docs: https://docs.python.org/3/library/random.html#random.choices
            # Our population will be the list of pages from our transition model,
            # the weights will be the values from our transition model
            # This function returns a list, so we only grab the first element
            sample = random.choices([page for page in sample_transition], [sample_transition[page]
                                    for page in sample_transition])[0]
            
        # Update the running count / probability in our output for the selected sample
        output[sample] = output[sample] + 1 / n

    return output
    # raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Set up our output dictionary and initialize values to 1 / count of all web pages
    output = {}
    web_pages_count = len(corpus)
    for web_page in corpus:
        # This is our baseline probability value
        output[web_page] = 1 / web_pages_count

    # Iterate through the probability distribution
    while True:
        # Create a deep copy of the existing probability distribution (since we will modify this with each iteration and compare the values afterwards)
        output_copy = copy.deepcopy(output)
        for current_page in output_copy:
            # Sum of probabilities to visit the current_page from the other pages
            prob_visit_from_other = 0

            # Iterate through all pages to identify which pages link to current_page
            for web_page in corpus:
                # Check if web_page links to other pages
                if not corpus[web_page] == set():
                    # Check if web_page links to current_page
                    if current_page in corpus[web_page]:
                        # If so, we add the probability of visiting the current_page from web_page
                        # This is equal to the probability that we are in web_page divided by the number of links from the web_page
                        prob_visit_from_other = prob_visit_from_other + output_copy[web_page] / len(corpus[web_page])
                else:
                    # Otherwise, the probability of visiting current_page from web_page is divided equally among all pages
                    prob_visit_from_other = prob_visit_from_other + output_copy[web_page] / web_pages_count

            # We update the distribution for current_page using our iteration formula
            output[current_page] = (1 - damping_factor) / web_pages_count + damping_factor * (prob_visit_from_other)
        
        # Check if we reached a covergence by comparing the previous values with the new values
        # We can do this via list comprehension for the difference in values
        # Basically, we are creating a list of the difference between the previous and new values only if such difference is greater than 0.001
        significant_differences = [output[page] - output_copy[page]
                                   for page in output_copy if abs(output[page] - output_copy[page]) > 0.001]
        
        # If there are no items in significant_difference, then we are done
        if len(significant_differences) == 0:
            break

    return output
    # raise NotImplementedError


if __name__ == "__main__":
    main()
