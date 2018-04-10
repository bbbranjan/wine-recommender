import pandas as pd

df = pd.read_csv('./wine-reviews/popular_terms.csv')
def get_popular_terms(query=''):
    # print(df)
    term_dict = {}
    for index, row in df.iterrows():
        term_dict[row['query']] = row['count']
    query = query.strip()

    if len(query) == 0:
        pass
    elif query.find(' ') == -1:
        print(query)
        if term_dict.has_key(query):
            term_dict[query] += 1
        else:
            term_dict[query] = 1
    else:
        for term in query.split(' '):
            print(term)
            if term_dict.has_key(term):
                term_dict[term] += 1
            else:
                term_dict[term] = 1

    updated_df = pd.DataFrame(term_dict.items(), columns=['query', 'count'])
    updated_df.to_csv('./wine-reviews/popular_terms.csv')

    desc_list = sorted(term_dict, key=term_dict.get, reverse=True)
    popular_terms_dict = {}
    for i in range (0,3):
        popular_terms_dict[desc_list[i]] = term_dict[desc_list[i]]
    return popular_terms_dict


if __name__ == '__main__':
    popularTerms()

