from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import time
import json


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        'Accept-Language': "en-US"

    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        # TODO
        self.crawling_threshold = crawling_threshold
        self.not_crawled = []
        self.crawled = []
        self.added_ids = []
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        return URL.split('/')[2]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        with open('IMDB_crawled.json', 'w') as f:
            json.dump(self.crawled, f)
        with open('IMDB_not_crawled.json', 'w') as f:
            json.dump(self.not_crawled, f) 
        with open('IMDB_added.json', 'w') as f:
            json.dump(self.added_ids, f)           



    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = json.load(f)

        with open('IMDB_not_crawled.json', 'r') as f:
            self.not_crawled = json.load(f)

        with open('IMDB_added.json', 'r') as f:
            self.added_ids = json.load(f)    


    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        try:
            response = get(URL, headers=self.headers)
            i = 0
            while response.status_code != 200 and i < 10:
                i+=1
                time.sleep(i)
                print("fail request ")
                response = get(URL, headers=self.headers)
    
            if response.status_code != 200:
               print("failed to crawl")
            return response
        except:
            print("failed to crawl")
            return None

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        # TODO update self.not_crawled and self.added_ids
        response = self.crawl(self.top_250_URL)
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            movies = soup.find_all('a', class_='ipc-title-link-wrapper')
            for i in range(250):
                link = movies[i]['href']
                movie_id = self.get_id_from_URL(link)
                self.not_crawled.append(movie_id)
                self.added_ids.append(movie_id)


    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO: 
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """
        print(len(self.crawled))
        if len(self.crawled)==0:
            self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=9) as executor:
          while crawled_counter < self.crawling_threshold:
            self.add_queue_lock.acquire()
            URL =  self.not_crawled.pop(0)
            futures.append(executor.submit(self.crawl_page_info, URL))
            crawled_counter += 1
            if len(self.not_crawled) == 0:
                self.add_queue_lock.release()
                wait(futures)
                futures = []
            else:    
                self.add_queue_lock.release()


        print('end pool')    


    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        print("new iteration " + URL)
        URL = "https://www.imdb.com/title/"+ URL +"/"
        response = self.crawl(URL)
        if response:
            movie = self.get_imdb_instance()
            self.extract_movie_info(response, movie, URL)

            try:
                self.add_queue_lock.acquire()
                self.crawled.append(movie)
                for link in movie['related_links']:
                    if self.get_id_from_URL(link) not in self.added_ids:
                        self.not_crawled.append(self.get_id_from_URL(link))
                        self.added_ids.append(self.get_id_from_URL(link))
            finally:             
                self.add_queue_lock.release()        
        print("end " + URL)

        

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        soup = BeautifulSoup(res.text, 'html.parser') 
        movie['id'] = URL.split('/')[4]
        movie['title'] = self.get_title(soup)
        movie['first_page_summary'] = self.get_first_page_summary(soup)
        movie['release_year'] = self.get_release_year(soup)
        movie['mpaa'] = self.get_mpaa(soup)
        movie['budget'] = self.get_budget(soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(soup)
        s2 = BeautifulSoup(self.crawl(self.get_summary_link(URL)).text, 'html.parser')
        movie['directors'] = self.get_director(soup)
        movie['writers'] = self.get_writers(soup)
        movie['stars'] = self.get_stars(soup)
        movie['related_links'] = self.get_related_links(soup)
        movie['genres'] = self.get_genres(soup)
        movie['languages'] = self.get_languages(soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(soup)
        movie['rating'] = self.get_rating(soup)
        movie['summaries'] = self.get_summary(s2)
        movie['synopsis'] = self.get_synopsis(s2)
        movie['reviews'] = self.get_reviews_with_scores(BeautifulSoup(self.crawl(self.get_review_link(URL)).text, 'html.parser'))

    def get_summary_link(self,url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            return url + 'plotsummary'
        except:
            print("failed to get summary link")

    def get_review_link(self,url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            return url + 'reviews'
        except:
            print("failed to get review link")

    def get_title(self,soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            return soup.find('span',class_='hero__primary-text').text
        except:
            print("failed to get title")

    def get_first_page_summary(self,soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            return soup.find('span', class_='sc-466bb6c-0 hlbAws').text

        except:
            print("failed to get first page summary")

    def get_director(self,soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            dirs = []
            list_dir = soup.find('div',class_='sc-67fa2588-3 fZhuJ').find('ul',class_= 'ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content baseAlt')
            for li in list_dir.find_all('li'):
                dirs.append(li.find('a').text)
            return dirs
        except:
            print("failed to get director")

    def get_stars(self,soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            stars = []
            list_stars = soup.find('div',class_='sc-67fa2588-3 fZhuJ').find_all('ul',class_= 'ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content baseAlt')[2]
            for li in list_stars.find_all('li'):
                stars.append(li.find('a').text)
            return stars    
        except:
            print("failed to get stars")

    def get_writers(self,soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            writers = []
            list_writers = soup.find('div',class_='sc-67fa2588-3 fZhuJ').find_all('ul',class_= 'ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content baseAlt')[1]
            for li in list_writers.find_all('li'):
                writers.append(li.find('a').text)
            return writers    
        except:
            print("failed to get writers")

    def get_related_links(self,soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            
            rls = soup.find('section',{'data-testid':'MoreLikeThis'}).find_all('a', class_= "ipc-poster-card__title ipc-poster-card__title--clamp-2 ipc-poster-card__title--clickable")
            links = []
            for rl in rls:
                links.append(rl['href'])
            return links
               
        except:
            print("failed to get related links")

    def get_summary(self,soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            ul = soup.find('div', {'data-testid':'sub-section-summaries'})
            summs = ul.find_all('div', class_='ipc-html-content-inner-div')
            summaries = []
            for summ in summs:
                summaries.append(summ.text)
            return summaries    
        except Exception as e:
            print("failed to get summaries:",e)

    def get_synopsis(self,soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            syns = soup.find('div', {'data-testid':'sub-section-synopsis'}).find_all('div', class_='ipc-html-content-inner-div')
            synopsis = []
            for syn in syns:
                synopsis.append(syn.text)
            return synopsis    

        except Exception as e:
            print("failed to get synopsis:",e)

    def get_reviews_with_scores(self,soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            sc = [] 
            lister_list = soup.find('div', class_= 'lister-list')
            rs = lister_list.find_all('div', class_='review-container')
            for r in rs:
                score = ''
                try:
                    score = r.find('span', class_='rating-other-user-rating').find('span').text
                except:
                    pass
                rev = ''
                try:
                    rev = r.find('div', class_='text show-more__control').text
                except:
                    pass
                sc.append([rev, score])
            return sc    
            
        except:
            print("failed to get reviews")

    def get_genres(self,soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            genres = []
            list_genres = soup.find('div',{'data-testid':'genres'},class_='ipc-chip-list--baseAlt ipc-chip-list').find_all('a') 
            #list_genres = soup.find('ul',class_='ipc-metadata-list ipc-metadata-list--dividers-all sc-ab73d3b3-1 jzyXiW ipc-metadata-list--base').find('li',{'data-testid':'storyline-genres'}).find_all('a')
            for li in list_genres:
                genres.append(li.text)
            return genres

        except:
            print("Failed to get generes")

    def get_rating(self,soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            return soup.find('div', class_='sc-bde20123-2 cdQqzc').find('span').text
        except:
            print("failed to get rating")

    def get_mpaa(self,soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            #cert = soup.find('ul',class_='ipc-metadata-list ipc-metadata-list--dividers-all sc-ab73d3b3-1 jzyXiW ipc-metadata-list--base').find('li',{'data-testid':'storyline-certificate'}).find('span', class_='ipc-metadata-list-item__list-content-item').text
            cert = soup.find('ul',class_ = 'ipc-inline-list ipc-inline-list--show-dividers sc-d8941411-2 cdJsTz baseAlt').find_all('a')[1].text
            return cert
        except:
            print("failed to get mpaa")

    def get_release_year(self,soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            ry = soup.find('ul',class_='ipc-inline-list ipc-inline-list--show-dividers sc-d8941411-2 cdJsTz baseAlt').find('a').text
            return ry
        except :
            print("failed to get release year")

    def get_languages(self,soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            lns = []
            list_langs = soup.find('li', {'data-testid':'title-details-languages'}).find_all('a')
            for li in list_langs:
                lns.append(li.text)
            return lns    
        except:
            print("failed to get languages")
            return None

    def get_countries_of_origin(self,soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            coo = []
            lisr_co = soup.find('li', {'data-testid':'title-details-origin'}).find_all('a')
            for li in lisr_co:
                coo.append(li.text)
            return coo    
        except:
            print("failed to get countries of origin")

    def get_budget(self,soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            budget = soup.find('li',{'data-testid':'title-boxoffice-budget'}).find('span', class_='ipc-metadata-list-item__list-content-item').text
            return budget
        except:
            print("failed to get budget")

    def get_gross_worldwide(self,soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            gross = soup.find('li',{'data-testid':'title-boxoffice-cumulativeworldwidegross'}).find('span', class_='ipc-metadata-list-item__list-content-item').text
            return gross
        except:
            print("failed to get gross worldwide")


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=10)
    imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    print("writing on file")
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
