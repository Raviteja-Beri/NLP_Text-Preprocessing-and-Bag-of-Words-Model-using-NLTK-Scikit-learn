import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

content = """India, a land of vibrant cultural diversity and rapid economic growth, 
exemplifies a unique blend of tradition and modernity. 
Its culture is a rich tapestry woven from ancient traditions, 
diverse languages, religions, and art forms. Festivals like Diwali and Holi, 
classical music, dance forms such as Bharatanatyam, 
and a culinary heritage spanning spicy curries to sweet delicacies reflect India's pluralistic identity. 
The economy, one of the world’s fastest-growing, is projected to be the third-largest by 2030, 
driven by sectors like technology, manufacturing, and services. 
Initiatives like "Make in India" and digital transformation through programs like,
Digital India have boosted foreign investment and innovation. India’s youthful workforce, 
with a median age of 28, fuels its economic dynamism, 
while reforms in taxation and infrastructure enhance global competitiveness. 
Despite challenges like income inequality and rural development, 
India’s commitment to sustainable growth and cultural preservation remains steadfast. 
Its global influence, rooted in its cultural soft power and economic potential, 
continues to rise, positioning India as a pivotal player in the 21st-century global landscape, 
balancing heritage with progress."""

# Process text
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(content)

corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer()
new_cv = cv.fit_transform(corpus).toarray()

print("Corpus:", corpus)
print("Vocabulary:", cv.get_feature_names_out())
print("Bag of Words Matrix:\n", new_cv)
