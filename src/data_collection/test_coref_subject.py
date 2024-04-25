
# # import packages
# import spacy
# from spacy.tokens import Doc, Span
# from typing import List  # Import List from typing
# # from fastcoref import FCoref

# def get_span_noun_indices(doc: Doc, cluster: List[List[int]]) -> List[int]:
#     spans = [doc[span[0]:span[1]+1] for span in cluster]
#     spans_pos = [[token.pos_ for token in span] for span in spans]
#     span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
#         if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
#     return span_noun_indices

# def core_logic_part(document, coref, resolved, mention_span):
#     # 'coref' is the current coreference mention [start, end] indices
#     # 'mention_span' is the Span object for the head entity
#     start, end = coref
#     # Replace the coreference mention in the resolved list with the head entity text
#     resolved[start] = mention_span.text + document[end+1].whitespace_
#     # Clear the text between the start and end of the mention
#     for i in range(start + 1, end + 1):
#         resolved[i] = ""


# def get_cluster_head(doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
#     head_idx = noun_indices[0]
#     head_start, head_end = cluster[head_idx]
#     head_span = doc[head_start:head_end+1]
#     return head_span, [head_start, head_end]

# def is_containing_other_spans(span: List[int], all_spans: List[List[int]]):
#     return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])

# def improved_replace_corefs(document, clusters):
#     print("Detected clusters:", clusters)  # Debug print
#     resolved = list(tok.text_with_ws for tok in document)
#     all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

#     print("All spans:", all_spans)  # Debug print

#     for cluster in clusters:
#         noun_indices = get_span_noun_indices(document, cluster)
#         print("Noun indices in cluster:", noun_indices)  # Debug print

#         if noun_indices:
#             mention_span, mention = get_cluster_head(document, cluster, noun_indices)
#             print(f"Head mention span and mention indices: {mention_span.text}, {mention}")  # Debug print

#             for coref in cluster:
#                 if coref != mention and not is_containing_other_spans(coref, all_spans):
#                     print(f"Replacing: {document[coref[0]:coref[1]+1].text} with {mention_span.text}")  # Debug print
#                     core_logic_part(document, coref, resolved, mention_span)

#     return "".join(resolved)



# def get_fast_cluster_spans(doc, clusters):
#     fast_clusters = []
#     for cluster in clusters:
#         new_group = []
#         for tuple in cluster:
#             print(type(tuple), tuple)
#             (start, end) = tuple
#             print("start, end", start, end)
#             span = doc.char_span(start, end)
#             print('span', span.start, span.end)
#             new_group.append([span.start, span.end-1])
#         fast_clusters.append(new_group)
#     return fast_clusters

# def get_fastcoref_clusters(doc, text):
#     preds = model.predict(texts=[text])
#     fast_clusters = preds[0].get_clusters(as_strings=False)
#     fast_cluster_spans = get_fast_cluster_spans(doc, fast_clusters)
#     return fast_cluster_spans
# # instantiate nlp and model objects
# nlp = spacy.load('en_core_web_sm')
# model = FCoref()

# # sample text to transform
# text = "Alice drove her car. She parked it outside the office."

# # rewrite the text with optimized FastCoref in three simple steps
# doc = nlp(text) # Step 1: Apply Spacy NLP model to create the doc
# clusters = get_fastcoref_clusters(doc, text) # Step 2: pass the Spacy doc and the text itself to get the FastCoref clusters AND convert them to the same annotation as AllenNLP
# coref_text = improved_replace_corefs(doc, clusters) # Step 3: pass the doc and the converted clusters to the NeuroSys function (provided above)

# print(text)
# print(coref_text)

import spacy
import spacy_experimental
import re
import pandas as pd

class coref_resolution:
  def __init__(self,text):
    self.text = text
  
  def get_coref_clusters(self,):
    self.nlp = spacy.load("en_core_web_trf")
    nlp_coref = spacy.load("en_coreference_web_trf")

    nlp_coref.replace_listeners("transformer", "coref", ["model.tok2vec"])
    nlp_coref.replace_listeners("transformer", "span_resolver", ["model.tok2vec"])

    self.nlp.add_pipe("coref", source=nlp_coref)
    self.nlp.add_pipe("span_resolver", source=nlp_coref)

    self.doc = self.nlp(self.text)
    self.tokens = [str(token) for token in self.doc]
    coref_clusters = {key : val for key , val in self.doc.spans.items() if re.match(r"coref_clusters_*",key)}

    return coref_clusters
  
  def find_span_start_end(self,coref_clusters):

    cluster_w_spans = {}
    for cluster in coref_clusters:
      cluster_w_spans[cluster] = [(span.start, span.end, span.text) for span in coref_clusters[cluster]]
    
    return cluster_w_spans
  
  def find_person_start_end(self, coref_clusters,cluster_w_spans):
    # nlp = spacy.load("en_core_web_trf")
    coref_clusters_with_name_spans = {}
    for key, val in coref_clusters.items():
      temp = [0 for i in range(len(val))]
      person_flag = False
      for idx, text in enumerate(val):
        doc = self.nlp(str(text))
        for word in doc.ents:
          # find the absolute token position of PERSON entity
          if word.label_ == 'PERSON':
            temp[idx] = (word.start, word.end, word.text)
            person_flag = True
        for token in doc:
          # find the absolute token position of the pronouns and references
          if token.pos_ == 'PRON':
            temp[idx] = (token.i,token.i+1,token)
      if len(temp) > 0:
        # replace the absolute token positions with the relative token positions in the entire corpus
        if person_flag:
          orig = cluster_w_spans[key]
          for idx, tup in enumerate(orig):
            if isinstance(tup, tuple) and isinstance(temp[idx], tuple):
              orig_start, orig_end, text = tup
              offset_start, offset_end, _ = temp[idx]
              orig_start += offset_start
              orig_end = orig_start + (offset_end - offset_start) 
              orig[idx] = (orig_start, orig_end, text)
          coref_clusters_with_name_spans[key] = orig

    return coref_clusters_with_name_spans
  
  def replace_refs_w_names(self,coref_clusters_with_name_spans):
    tokens = self.tokens
    special_tokens = ["my","his","her","mine"]
    for key, val in coref_clusters_with_name_spans.items():
      if len(val) > 0 and isinstance(val, list):
        head = val[0]
        head_start, head_end, _ = head
        head_name = " ".join(tokens[head_start:head_end])
        for i in range(1,len(val)):
          coref_token_start, coref_token_end, _ = val[i]
          count = 0
          for j in range(coref_token_start, coref_token_end):
            if tokens[j].upper() == "I":
                count += 1
                continue
            if count == 0:
              if tokens[j].lower() in special_tokens:
                if head_name[-1].lower() == "s":
                  tokens[j] = str(head_name)+"'"
                else:
                  tokens[j] = str(head_name)+"'s"
              else:
                tokens[j] = head_name
            else:
              tokens[j] = ""
            count += 1

    return tokens
  
  def main(self,):

    coref_clusters = self.get_coref_clusters()
    coref_w_spans = self.find_span_start_end(coref_clusters)
    coref_clusters_with_name_spans = self.find_person_start_end(coref_clusters,coref_w_spans)
    tokens = self.replace_refs_w_names(coref_clusters_with_name_spans)

    return " ".join(tokens)

text = """John took music class even though he resented it. His interest was more in science and tech"""
obj = coref_resolution(text)
coref_clusters = obj.get_coref_clusters()
print(coref_clusters)
