import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import nest_asyncio
import re
import requests
from io import BytesIO
# Apply nest_asyncio to allow nested event loops (useful for Jupyter)
nest_asyncio.apply()

# Load a pre-trained sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Asynchronous function to fetch paper abstract
async def fetch_paper_abstract(session, paper_url,is_acl=True):
    try:
        async with session.get(paper_url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                if is_acl:
                    # ACL-specific abstract extraction
                    all_spans = soup.find_all("span")
                    abstract = " ".join([span.text.strip() for span in all_spans if not span.has_attr('class')])
                else:
                    # ECCV-specific abstract extraction
                    abstract_div = soup.find('div', id='abstract')
                    abstract = abstract_div.text.strip() if abstract_div else 'Abstract not available'
                
                return abstract
            return None
    except Exception:
        return None

# Asynchronous function to fetch ACL papers
async def acl_papers(year):
    url = f"https://aclanthology.org/events/acl-{year}/"
    titles, abstracts, abs_links, pdf_links = [], [], [], []

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                for paper_entry in soup.find_all('p', class_='d-sm-flex align-items-stretch'):
                    title_element = paper_entry.find('strong')
                    if title_element:
                        title = title_element.find_next('a').text
                        abs_link = "https://aclanthology.org/" + title_element.find_next('a').get('href')

                        if "https://github.com/baidu" not in abs_link and "pdf\n" not in title:
                            titles.append(title)
                            abs_links.append(abs_link)
                            pdf_link_element = paper_entry.find('span', class_="d-block mr-2 text-nowrap list-button-row")
                            pdf_link = pdf_link_element.find('a').get('href') if pdf_link_element else None
                            pdf_links.append("https://aclanthology.org/" + pdf_link if pdf_link else None)

        tasks = [fetch_paper_abstract(session, link, is_acl=True) for link in abs_links]
        abstracts = await asyncio.gather(*tasks)

    return pd.DataFrame({'title': titles, 'abstract': abstracts, 'pdf_link': pdf_links})

# Asynchronous function to fetch ECCV papers
async def eccv_papers(year):
    year = f"eccv_{year}"
    paper_titles, paper_links, paper_pdfs, paper_abstracts = [], [], [], []

    url = "https://www.ecva.net/papers.php"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                titles = soup.find_all('div', class_='accordion-content')
                for title in titles:
                    for anchor in title.find_all('a'):
                        href = anchor.get('href')
                        if href and href.endswith('.php') and year in href:
                            paper_titles.append(anchor.text.strip())
                            full_url = f'https://www.ecva.net/{href}'
                            paper_links.append(full_url)
                            pdf_link = full_url.replace('.php', '.php')
                            paper_pdfs.append(pdf_link)

        tasks = [fetch_paper_abstract(session, link, is_acl=False) for link in paper_links]
        paper_abstracts = await asyncio.gather(*tasks)

    return pd.DataFrame({'title': paper_titles, 'abstract': paper_abstracts, 'pdf_link': paper_pdfs})

# Wrapper functions
def run_acl_papers(year):
    return asyncio.run(acl_papers(year))

def run_eccv_papers(year):
    return asyncio.run(eccv_papers(year))

# Embed papers and create FAISS index
def embed_papers(df):
    df['abstract'] = df['abstract'].astype(str)
    combined_texts = (df['title'] + ' ' + df['abstract']).apply(lambda text: re.match(r'(\S+\s*){1,30}', text).group(0))
    embeddings = model.encode(combined_texts.tolist(), convert_to_tensor=False)
    return embeddings

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Query function
def query_papers(query, index, df, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = (distances[0] - distances[0].min()) / (distances[0].max() - distances[0].min())
    return results.head(top_k)

async def fetch_paper_abstract_and_pdf(session, link):
    try:
        async with session.get(link) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                
                # Extract abstract
                abstract_div = soup.find('div', id='abstractExample')
                abstract = abstract_div.find('p').text.strip() if abstract_div and abstract_div.find('p') else 'Abstract not available'

                # Extract PDF link
                pdf_link = None
                pdf_tag = soup.find('a', href=lambda x: x and '.pdf' in x)
                if pdf_tag:
                    pdf_link = pdf_tag.get('href')
                    if not pdf_link.startswith('http'):
                        pdf_link = "https://iclr.cc" + pdf_link  # Make the URL absolute if it's relative
                return abstract, pdf_link
            else:
                return 'Abstract not available', None
    except Exception:
        return 'Error fetching abstract', None

# Asynchronous function to fetch ICLR papers
async def iclr_papers_with_pdf(year):
    titles, links, abstracts, pdf_links = [], [], [], []
    url = f"https://iclr.cc/virtual/{year}/papers.html?filter=titles"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    title_list = soup.find('ul', class_="nav nav-pills")
    titles_data = title_list.find_next('ul').find_all('a')

    for head in titles_data:
        titles.append(head.text.strip())
        full_url = "https://iclr.cc/" + head.get('href')
        links.append(full_url)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_paper_abstract_and_pdf(session, link) for link in links]
        paper_data = await asyncio.gather(*tasks)

        for abstract, pdf_link in paper_data:
            abstracts.append(abstract)
            pdf_links.append(pdf_link)

    return pd.DataFrame({'title': titles, 'abstract': abstracts, 'pdf_link': links})

# Wrapper for asynchronous ICLR paper fetching
def run_iclr_papers_with_pdf(year):
    try:
        return asyncio.run(iclr_papers_with_pdf(year))
    except RuntimeError:
        return asyncio.get_event_loop().run_until_complete(iclr_papers_with_pdf(year))

# Embedding function
def embed_papers(df):
    combined_texts = (df['title'] + ' ' + df['abstract']).apply(lambda text: re.match(r'(\S+\s*){1,30}', text).group(0))
    embeddings = model.encode(combined_texts.tolist(), convert_to_tensor=False)
    return embeddings

# FAISS index creation function
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Query function for the FAISS index
def query_papers(query, index, df, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = (distances[0] - distances[0].min()) / (distances[0].max() - distances[0].min())
    return results.head(top_k)


# Dictionary for ICML year and corresponding volume number
icml_volumes = {
    2016: 48,
    2017: 70,
    2018: 80,
    2019: 97,
    2020: 119,
    2021: 139,
    2022: 162,
    2023: 202,
    2024: 250  # Hypothetical for 2024, update when available
}

# Asynchronous function to fetch a single paper's abstract
async def fetch_paper_abstract(session, link,is_acl):
    try:
        async with session.get(link) as response:
            if response.status == 200:
                html = await response.text()
                paper_soup = BeautifulSoup(html, "html.parser")
                
                # Attempt to extract abstract from known abstract locations in ICML
                abstract_div = paper_soup.find('div', {'id': 'abstract'})
                if abstract_div:
                    return abstract_div.text.strip()
                
                # Fallback if div with id='abstract' is not found
                return 'Abstract not available'
            else:
                return f'Error: {response.status}'
    except Exception as e:
        return f'Error fetching abstract: {e}'

# Function to fetch ICML papers for a given year
def icml_papers(year):
    paper_titles = []
    paper_links = []
    pdf_links = []
    paper_years = []
    abstracts = []

    # Get the volume number for the year
    volume = icml_volumes.get(year)
    if volume is None:
        print(f"Volume for ICML {year} is not available.")
        return None

    # URL of the ICML page for a specific year
    url = f"https://proceedings.mlr.press/v{volume}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all paper listings
    papers = soup.find_all('div', class_='paper')

    # Check if papers were found
    if not papers:
        print(f"No papers found for ICML {year} at {url}")
        return None

    # Loop over all papers
    for paper_div in papers:
        # Get the title from 'p' tag with class 'title'
        title = paper_div.find('p', class_='title').text.strip()
        
        # Get the paper link from 'a' tag (if available)
        paper_link = paper_div.find('a').get('href')
        if not paper_link.startswith('http'):
            paper_link = f"https://proceedings.mlr.press{paper_link}"

        # Get the PDF link (from 'a' tag with text 'Download PDF')
        pdf_link = paper_div.find('a', text="Download PDF")
        if pdf_link:
            pdf_link = pdf_link.get('href')
            if not pdf_link.startswith('http'):
                pdf_link = f"https://proceedings.mlr.press{pdf_link}"
        else:
            pdf_link = 'No PDF link available'

        # Append the title, link, PDF, and year
        paper_titles.append(title)
        paper_links.append(paper_link)
        pdf_links.append(pdf_link)
        paper_years.append(year)

    # Create DataFrame with scraped data
    df = pd.DataFrame({
        'year': paper_years,
        'title': paper_titles,
        'paper_link': paper_links,
        'pdf_link': pdf_links,
    })

    return df

# Function to create FAISS index for ICML papers
def create_faiss_index_icml(papers_df):
    titles = papers_df['title'].tolist()

    # Embed titles for FAISS indexing
    embeddings = model.encode(titles, show_progress_bar=True)

    # Normalize embeddings for cosine similarity
    normalized_embeddings = np.array([embedding / np.linalg.norm(embedding) for embedding in embeddings])

    # Initialize FAISS index with embedding dimension size
    dimension = normalized_embeddings.shape[1]  # Embedding size from the model
    faiss_index = faiss.IndexFlatIP(dimension)  # Using Inner Product (dot product, akin to cosine similarity)

    # Add embeddings to the FAISS index
    faiss_index.add(normalized_embeddings)

    return faiss_index, papers_df

# Query-based retrieval function with PDF links for ICML papers
def query_faiss_with_pdf_icml(faiss_index, query, papers_df, top_k=5):
    # Embed the query
    query_embedding = model.encode([query])[0]

    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Search the FAISS index
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)

    # Retrieve relevant papers
    results = papers_df.iloc[indices[0]].copy()
    results['similarity_score'] = distances[0]  # Add similarity scores to the result

    return results

# Function to fetch ICML papers and perform query-based retrieval
def run_query_based_retrieval_with_pdf_icml(year, query, top_k=5):
    # Fetch ICML papers for the given year
    papers_df = icml_papers(year)

    # Create FAISS index for the papers
    faiss_index, papers_df = create_faiss_index_icml(papers_df)

    # Fetch abstracts asynchronously for the papers
    async def fetch_all_abstracts(session):
        tasks = [fetch_paper_abstract(session, link,is_acl=False) for link in papers_df['paper_link']]
        return await asyncio.gather(*tasks)

    async def fetch_abstracts():
        async with aiohttp.ClientSession() as session:
            return await fetch_all_abstracts(session)

    # Run the async function to fetch abstracts
    abstracts = asyncio.run(fetch_abstracts())
    papers_df['abstract'] = abstracts

    # Perform query-based retrieval
    relevant_papers = query_faiss_with_pdf_icml(faiss_index, query, papers_df, top_k=top_k)

    # Save results to CSV
    relevant_papers.to_csv(f"relevant_icml_papers_with_pdf_{year}.csv", index=False)

    return relevant_papers


def get_all_papers(conf, year):
    data = []
    base_url = f"https://openaccess.thecvf.com/{conf}{year}"
    url = f"{base_url}?day=all"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    paper_titles = soup.find_all('dt', class_='ptitle')

    for title in paper_titles:
        paper_title = title.find('a').text.strip()
        a_tag = title.find('a')
        if a_tag:
            paper_link = a_tag.get('href')
            paper_link = f"http://openaccess.thecvf.com/{paper_link}"

            # Find corresponding PDF link
            pdf_link = None
            pdf_tag = title.find_next_sibling('dd')
            if pdf_tag and pdf_tag.find('a'):
                pdf_link = pdf_tag.find('a').get('href')
                if pdf_link.endswith('.pdf'):
                    pdf_link = f"http://openaccess.thecvf.com/{pdf_link}"

            # Add the data to the list
            data.append({
                "year": year,
                "title": paper_title,
                "link": paper_link,
                "pdf_link": pdf_link,
                "abstract": None,  # Abstract will be fetched asynchronously
                "Conference": conf
            })
    return data

# Asynchronous function to fetch abstract for a given paper link
async def fetch_abstract_iccv(session, link):
    async with session.get(link) as response:
        text = await response.text()
        soup = BeautifulSoup(text, 'html.parser')
        abstract_tag = soup.find('div', id='abstract')
        if abstract_tag:
            return abstract_tag.text.strip()
        else:
            return "No Abstract Available"

# Function to fetch abstracts asynchronously
async def get_abstracts_iccv(data):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_abstract(session, paper['link']) for paper in data]
        abstracts = await asyncio.gather(*tasks)

    for i, paper in enumerate(data):
        paper['abstract'] = abstracts[i]
    return data

# Fetch papers and their abstracts for a given conference and year
def fetch_conference_papers_iccv(conf, year):
    data = get_all_papers(conf, year)
    loop = asyncio.get_event_loop()
    data = loop.run_until_complete(get_abstracts(data))

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# Function to create FAISS index for the conference papers
def create_faiss_index_conference_iccv(papers_df):
    # Combine titles and abstracts for embedding
    combined_text = (papers_df['title'] + " " + papers_df['abstract']).tolist()

    # Embed combined text for FAISS indexing
    embeddings = model.encode(combined_text, show_progress_bar=True)

    # Normalize embeddings for cosine similarity
    normalized_embeddings = np.array([embedding / np.linalg.norm(embedding) for embedding in embeddings])

    # Initialize FAISS index with embedding dimension size
    dimension = normalized_embeddings.shape[1]  # Embedding size from the model
    faiss_index = faiss.IndexFlatIP(dimension)  # Using Inner Product (dot product, akin to cosine similarity)

    # Add embeddings to the FAISS index
    faiss_index.add(normalized_embeddings)

    return faiss_index, papers_df

# Query-based retrieval function with PDF links for conference papers
def query_faiss_with_pdf_conference_iccv(faiss_index, query, papers_df, top_k=5):
    # Embed the query
    query_embedding = model.encode([query])[0]

    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Search the FAISS index
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)

    # Retrieve relevant papers
    results = papers_df.iloc[indices[0]].copy()
    results['similarity_score'] = distances[0]  # Add similarity scores to the result

    return results

# Function to run the entire process for conference paper retrieval
def run_query_based_retrieval_with_pdf_iccv(conf, year, query, top_k=5):
    # Fetch papers for the given conference and year
    papers_df = fetch_conference_papers_iccv(conf, year)

    # Create FAISS index for the papers
    faiss_index, papers_df = create_faiss_index_conference_iccv(papers_df)

    # Perform query-based retrieval
    relevant_papers = query_faiss_with_pdf_conference_iccv(faiss_index, query, papers_df, top_k=top_k)

    # Save results to CSV
    relevant_papers.to_csv(f"relevant_{conf}_papers_with_pdf_{year}.csv", index=False)

    return relevant_papers

def get_all_papers_cvpr(conf, year):
    data = []
    base_url = f"https://openaccess.thecvf.com/{conf}{year}"
    url = f"{base_url}?day=all"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    paper_titles = soup.find_all('dt', class_='ptitle')

    for title in paper_titles:
        paper_title = title.find('a').text.strip()
        a_tag = title.find('a')
        if a_tag:
            paper_link = a_tag.get('href')
            paper_link = f"http://openaccess.thecvf.com/{paper_link}"

            # Find corresponding PDF link
            pdf_link = None
            pdf_tag = title.find_next_sibling('dd')
            if pdf_tag and pdf_tag.find('a'):
                pdf_link = pdf_tag.find('a').get('href')
                if pdf_link.endswith('.pdf'):
                    pdf_link = f"http://openaccess.thecvf.com/{pdf_link}"

            # Add the data to the list
            data.append({
                "year": year,
                "title": paper_title,
                "link": paper_link,
                "pdf_link": pdf_link,
                "abstract": None,  # Abstract will be fetched asynchronously
                "Conference": conf
            })
    return data

# Asynchronous function to fetch abstract for a given paper link
async def fetch_abstract_cvpr(session, link):
    async with session.get(link) as response:
        text = await response.text()
        soup = BeautifulSoup(text, 'html.parser')
        abstract_tag = soup.find('div', id='abstract')
        if abstract_tag:
            return abstract_tag.text.strip()
        else:
            return "No Abstract Available"

# Function to fetch abstracts asynchronously
async def get_abstracts(data):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_abstract_cvpr(session, paper['link']) for paper in data]
        abstracts = await asyncio.gather(*tasks)

    for i, paper in enumerate(data):
        paper['abstract'] = abstracts[i]
    return data

# Fetch papers and their abstracts for a given conference and year
def fetch_conference_papers_cvpr(conf, year):
    data = get_all_papers_cvpr(conf, year)
    loop = asyncio.get_event_loop()
    data = loop.run_until_complete(get_abstracts(data))

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# Function to create FAISS index for the conference papers
def create_faiss_index_conference_cvpr(papers_df):
    # Combine titles and abstracts for embedding
    combined_text = (papers_df['title'] + " " + papers_df['abstract']).tolist()

    # Embed combined text for FAISS indexing
    embeddings = model.encode(combined_text, show_progress_bar=True)

    # Normalize embeddings for cosine similarity
    normalized_embeddings = np.array([embedding / np.linalg.norm(embedding) for embedding in embeddings])

    # Initialize FAISS index with embedding dimension size
    dimension = normalized_embeddings.shape[1]  # Embedding size from the model
    faiss_index = faiss.IndexFlatIP(dimension)  # Using Inner Product (dot product, akin to cosine similarity)

    # Add embeddings to the FAISS index
    faiss_index.add(normalized_embeddings)

    return faiss_index, papers_df

# Query-based retrieval function with PDF links for conference papers
def query_faiss_with_pdf_conference_cvpr(faiss_index, query, papers_df, top_k=5):
    # Embed the query
    query_embedding = model.encode([query])[0]

    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Search the FAISS index
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)

    # Retrieve relevant papers
    results = papers_df.iloc[indices[0]].copy()
    results['similarity_score'] = distances[0]  # Add similarity scores to the result

    return results

# Function to run the entire process for conference paper retrieval
def run_query_based_retrieval_with_pdf_cvpr(conf, year, query, top_k=5):
    # Fetch papers for the given conference and year
    papers_df = fetch_conference_papers_cvpr(conf, year)

    # Create FAISS index for the papers
    faiss_index, papers_df = create_faiss_index_conference_cvpr(papers_df)

    # Perform query-based retrieval
    relevant_papers = query_faiss_with_pdf_conference_cvpr(faiss_index, query, papers_df, top_k=top_k)

    # Save results to CSV
    relevant_papers.to_csv(f"relevant_{conf}_papers_with_pdf_{year}.csv", index=False)

    return relevant_papers

async def fetch_with_retries_nips(session, url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                text = await response.text()
                return text
        except aiohttp.ClientConnectionError as e:
            print(f"Connection error: {e}, retrying... ({attempt + 1}/{retries})")
        except aiohttp.ClientResponseError as e:
            print(f"Response error: {e}, retrying... ({attempt + 1}/{retries})")
        except asyncio.TimeoutError:
            print(f"Timeout on {url}, retrying... ({attempt + 1}/{retries})")

    raise aiohttp.ServerDisconnectedError(f"Failed to fetch {url} after {retries} retries")

# Asynchronous function to fetch a single paper's title, abstract, and PDF link
async def fetch_paper_data_nips(session, url):
    try:
        text = await fetch_with_retries_nips(session, url)
        soup = BeautifulSoup(text, "html.parser")

        # Get the paper title
        title_tag = soup.find('title')
        paper_title = title_tag.text.strip() if title_tag else "No Title"

        # Replace abstract extraction logic with your code
        # Find the 'Abstract' heading, then get the next paragraph containing the actual abstract
        abstract_heading = soup.find('p')
        if abstract_heading:
            abstract_tag = abstract_heading.find_next('p').find_next('p')
            abstract = abstract_tag.text if abstract_tag else "No Abstract"
        else:
            abstract = "No Abstract"

        # Generate the PDF link
        pdf_tag = soup.find('a', href=lambda x: x and x.endswith('.pdf'))
        pdf_link = pdf_tag['href'] if pdf_tag else "No PDF Link"

        if not pdf_link.startswith('http'):
            pdf_link = f"https://papers.nips.cc{pdf_link}"

        return paper_title, abstract, pdf_link
    except Exception as e:
        print(f"Failed to fetch paper data from {url}: {e}")
        return "No Title", "No Abstract", "No PDF Link"

# Function to fetch batch of papers
async def fetch_batch_papers_nips(session, batch):
    tasks = [fetch_paper_data_nips(session, url) for url in batch]
    paper_data = await asyncio.gather(*tasks)
    return paper_data

# Function to fetch all papers for the specified year in batches
async def fetch_all_papers_in_batches_nips(base_url, year, batch_size=10, delay_between_batches=5):
    url = f"{base_url}/{year}"
    async with aiohttp.ClientSession() as session:
        text = await fetch_with_retries_nips(session, url)
        soup = BeautifulSoup(text, "html.parser")
        titles = soup.find_all('a', href=True)

        # Prepare the URLs for the paper details
        paper_urls = [f"https://papers.nips.cc{title.get('href')}" for title in titles]

        # Split the paper URLs into batches
        batches = [paper_urls[i:i+batch_size] for i in range(0, len(paper_urls), batch_size)]

        all_paper_data = []

        for batch in batches:
            # Fetch papers in the current batch
            batch_paper_data = await fetch_batch_papers_nips(session, batch)
            all_paper_data.extend(batch_paper_data)

            # Wait for a delay between batches to avoid server overload
            await asyncio.sleep(delay_between_batches)

        papers = [data[0] for data in all_paper_data]
        abstracts = [data[1] for data in all_paper_data]
        pdf_links = [data[2] for data in all_paper_data]

        df = pd.DataFrame({'Title': papers, 'Abstract': abstracts, 'PDF Link': pdf_links})
        return df

# Wrapper to run the asynchronous function
def nips_papers(year, batch_size=10):
    base_url = "https://papers.nips.cc/paper_files/paper"
    loop = asyncio.get_event_loop()
    df = loop.run_until_complete(fetch_all_papers_in_batches_nips(base_url, year, batch_size))
    return df

# Function to create FAISS index for NIPS papers
def create_faiss_index_nips(papers_df):
    titles = papers_df['Title'].tolist()
    abstracts = papers_df['Abstract'].tolist()

    # Combine titles and abstracts for embedding
    documents = [f"{title} {abstract}" for title, abstract in zip(titles, abstracts)]

    # Embed documents
    embeddings = model.encode(documents, show_progress_bar=True)

    # Normalize embeddings for cosine similarity
    normalized_embeddings = np.array([embedding / np.linalg.norm(embedding) for embedding in embeddings])

    # Initialize FAISS index with embedding dimension size
    dimension = normalized_embeddings.shape[1]  # Embedding size from the model
    faiss_index = faiss.IndexFlatIP(dimension)  # Using Inner Product (dot product, akin to cosine similarity)

    # Add embeddings to the FAISS index
    faiss_index.add(normalized_embeddings)

    return faiss_index, papers_df

# Query-based retrieval function with PDF links for NIPS papers
def query_faiss_with_pdf_nips(faiss_index, query, papers_df, top_k=5):
    # Embed the query
    query_embedding = model.encode([query])[0]

    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Search the FAISS index
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)

    # Retrieve relevant papers
    results = papers_df.iloc[indices[0]]
    results['similarity_score'] = distances[0]  # Add similarity scores to the result

    return results

# Function to fetch NIPS papers and perform query-based retrieval
def run_query_based_retrieval_with_pdf_nips1(year, query, top_k=5):
    # Fetch NIPS papers for the given year
    papers_df = nips_papers(year, batch_size=2000)

    # Create FAISS index for the papers
    faiss_index, papers_df = create_faiss_index(papers_df)

    # Perform query-based retrieval
    relevant_papers = query_faiss_with_pdf_nips(faiss_index, query, papers_df, top_k=top_k)

    # Save results to CSV (or output in other formats like JSON/XLSX as needed)
    relevant_papers.to_csv(f"relevant_nips_papers_with_pdf_{year}.csv", index=False)

    return relevant_papers
# Streamlit Interface


st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("C:\\Users\\maury\\Downloads\\image1.jpg");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.title("Research Paper Search Engine")
conference = st.selectbox("Select Conference", ["ACL","ECCV","ICLR","ICML","NIPS","ICCV","CVPR"])
year = st.text_input("Enter Year", "2021")
query = st.text_input("Enter Query")

if st.button("Search"):
    with st.spinner("Fetching and indexing papers..."):
        if conference == "ACL":
            df_papers = run_acl_papers(year)
        elif conference == "ECCV":
            df_papers = run_eccv_papers(year)
        elif conference == "ICLR":
            df_papers = run_iclr_papers_with_pdf(year)
        elif conference == "ICML":
            results = run_query_based_retrieval_with_pdf_icml(int(year), query)
        elif conference == 'ICCV' :
            df_papers = run_query_based_retrieval_with_pdf_iccv('ICCV',year,query)
        elif conference == 'CVPR' :
             
            df_papers =run_query_based_retrieval_with_pdf_cvpr(conference, year, query, top_k=5)
        elif conference == 'NIPS' :
            df_papers=run_query_based_retrieval_with_pdf_nips1(year, query)  


    
    if conference != "ICML":
        paper_embeddings = np.array(embed_papers(df_papers))
        faiss_index = create_faiss_index(paper_embeddings)
        results = query_papers(query, faiss_index, df_papers, top_k=5)

    st.write("## Results")
    for _, row in results.iterrows():
        st.write(f"### {row['title']}")
        st.write(f"Abstract: {row['abstract']}")
        st.write(f"[PDF Link]({row['pdf_link']})")
        st.write(f"Similarity Score: {row['similarity_score']:.2f}")


    csv = results.to_csv(index=False)
    b = BytesIO()
    b.write(csv.encode())
    b.seek(0)

    # Download button for CSV
    st.download_button(
        label="Download results as CSV",
        data=b,
        file_name=conference+"_top_5_research_papers.csv",
        mime="text/csv"
    )