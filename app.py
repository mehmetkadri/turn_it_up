import json
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny import *
from shiny.types import FileInfo
import os
import re
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings('ignore')
import lemmatizer
import pickle
import nltk
from snowballstemmer import TurkishStemmer


choices = {"none": "", "stem": "Stemming", "lemma": "Lemmatization"}

app_ui = ui.page_fluid(

    ui.h1("turnitup"),
    #ui.p("Even better than turnitin.com"),

    ui.navset_tab_card(
        ui.nav("Document upload",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_file("fileLeft", "Choose the text file to be compared", accept=[
                        ".txt"], multiple=False),
                    ui.input_file("filesRight", "Choose the text file(s) to be compared with", accept=[
                        ".txt"], multiple=True),
                ),
                ui.panel_main(
                    ui.output_ui("uploadDocuments"),
                ),
            )
        ),

        ui.nav("Document comparison",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_checkbox("decapitalization", "Decapitalize"),
                    ui.input_checkbox("punctuation", "Remove Punctuations"),
                    ui.input_numeric("stopword", "Number of common stop-words to remove:", value=0),
                    ui.input_checkbox("synonym", "Check synonyms"),
                    ui.input_select("stem_lem", "Apply stemming or lemmatization", choices),
                    "---------------------------------------------------------------------",
                    "---------------------------------------------------------------------",
                    ui.input_checkbox("compare", "Ready to Compare"),
                ),
                ui.panel_main(
                    ui.output_ui("comparison"),
                ),
            ), value=0
        ),

        ui.nav_menu(
            "Results",
                ui.nav("Report",
                    ui.output_ui("report")
                ),
                "----",
                ui.nav("Show similiar parts of the documents",
                    ui.output_ui("similiar")
                ),
        ),
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    @output
    @render.ui
    def uploadDocuments():
        if input.fileLeft() is None:
            return "Please upload the text file to be compared."

        if input.fileLeft() is not None:
            empty_left()
            f: list[FileInfo] = input.fileLeft()
            with open(f[0]['datapath'], "r", encoding="UTF-8") as file:
                with open(os.path.join("./turn_it_up/uploads/left", f[0]['name']), "w", encoding="UTF-8") as left_file:
                    left_file.write(file.read())

        if input.filesRight() is None:
            return "The text file to be compared is successfully uploaded. Now, please upload the text file(s) to be compared to."

        if input.filesRight() is not None:
            empty_right()
            f: list[FileInfo] = input.filesRight()
            for i in range(len(f)):
                with open(f[i]['datapath'], "r", encoding="UTF-8") as file:
                    with open(os.path.join("./turn_it_up/uploads/right", f[i]['name']), "w", encoding="UTF-8") as right_file:
                        right_file.write(file.read())

        return "Both documents are successfully uploaded. Now, you can compare them."

    @output
    @render.ui
    @reactive.Calc
    def comparison():
        if input.fileLeft() is None or input.filesRight() is None:
            return "Please complete the upload process first."
        if input.compare():
            documents = {"left_files": [], "right_files": []}

            for i in range(len(os.listdir("./turn_it_up/uploads/right"))):
                a = os.listdir("./turn_it_up/uploads/left")[0]
                with open(os.path.join("./turn_it_up/uploads/left", a), "r", encoding="UTF-8") as file:
                    documents["left_files"].append({"name": a.split(".")[0], "content": file.read()})
            
            for i in range(len(os.listdir("./turn_it_up/uploads/right"))):
                b = os.listdir("./turn_it_up/uploads/right")[i]
                with open(os.path.join("./turn_it_up/uploads/right", b), "r", encoding="UTF-8") as file:
                    documents["right_files"].append({"name": b.split(".")[0], "content": file.read()})
            
            
            for i in range(len(documents["left_files"])):
                documents["left_files"][i]["content"] = re.split(r'(?<=[^A-Z].[.?!]) +(?=[A-Z])', documents["left_files"][i]["content"])

            for i in range(len(documents["right_files"])):
                documents["right_files"][i]["content"] = re.split(r'(?<=[^A-Z].[.?!]) +(?=[A-Z])', documents["right_files"][i]["content"])

            
            # decapitalize the sentences
            if input.decapitalization():
                documents = decapitalize(documents)

            # punctuation removal
            if input.punctuation():
                documents = remove_punctuation(documents)
            
            # stopword removal
            if input.stopword() > 0:
                documents, common_words = remove_stopwords(documents, input.stopword())
                with open("./turn_it_up/results/common_words.txt", "w", encoding="UTF-8") as file:
                    for i in range(len(common_words)):
                        file.write(documents["left_files"][i]["name"]+".txt and " + documents["right_files"][i]["name"]+".txt have most frequently used common words :" + str(common_words[i]) + "\n")
            else:
                if os.path.exists("./turn_it_up/results/common_words.txt"):
                    os.remove("./turn_it_up/results/common_words.txt")
            
            
            # synonym check
            if input.synonym():
                documents = check_synonym(documents)
            
            # stemming or lemmatization
            if input.stem_lem() == "stem":
                documents = stemming(documents)
            elif input.stem_lem() == "lemma":
                documents = lemmatization(documents)
            else:
                pass

            # similarity check
            assignments, overall_sim = classify(documents)

            # report
            with open("./turn_it_up/results/assignments.json", "w", encoding="UTF-8") as file:
                json.dump(assignments, file)
            with open("./turn_it_up/results/overall_sim.json", "w", encoding="UTF-8") as file:
                json.dump(overall_sim, file)

            output = ""
            output += "<h3>Preprocessed documents</h3>"
            output += "<table><tr><th style = 'padding: 15px;'>Name of the Compared Document</th><th style = 'padding: 15px;'> Content </th><th style = 'padding: 15px;'>Name of the Document to Compare</th><th style = 'padding: 15px;'>Content</th></tr>"
            for i in range(len(documents["left_files"])):
                output += "<tr><td style = 'padding: 15px;'>" + documents["left_files"][i]["name"] + "</td><td style = 'padding: 15px;'>" + "".join(documents["left_files"][i]["content"]) + "</td><td style = 'padding: 15px;'>" + documents["right_files"][i]["name"] + "</td><td style = 'padding: 15px;'>" + "".join(documents["right_files"][i]["content"]) + "</td></tr>"
            output += "</table>\n"

            return ui.HTML(output)
        else:
            return "Please select 'Ready to Compare' to compare the documents."
              
    @output
    @render.ui
    @reactive.Calc
    def report():
        output = ""
        if input.fileLeft() is None or input.filesRight() is None:
            return ui.HTML("<h3>Please complete the upload process first.</h3>")
        if input.compare():
            # read from assignments.json and overall_sim.json
            with open("./turn_it_up/results/assignments.json", "r", encoding="UTF-8") as file:
                assignments = json.load(file)
            with open("./turn_it_up/results/overall_sim.json", "r", encoding="UTF-8") as file:
                overall_sim = json.load(file)
            
            # generate report
            output += "<h3>Overall Similarity</h3>"
            output += "<p>The overall similarity between documents is as follows:</p>"
            for i in overall_sim:
                output += "<p>" + os.listdir("./turn_it_up/uploads/left")[0] + " and " + str(i) + ".txt : " + str(overall_sim[i]) + "</p>"
            
            output += "<h3>Similarity between each sentence</h3>"
            output += "<p>The similarity between each sentence that is higher than the limit is shown in the table below.</p>"
            
            # generate table
            output += "<table><tr><th style = 'padding: 15px;'>" + os.listdir("./turn_it_up/uploads/left")[0] + " Sentence Index</th><th style = 'padding: 15px;'> Sentence </th><th style = 'padding: 15px;'>Compared Document Name</th><th style = 'padding: 15px;'>Sentence Index</th><th style = 'padding: 15px;'> Sentence </th><th style = 'padding: 15px;'>Similarity</th></tr>"
            for i in assignments:
                with open(os.path.join("./turn_it_up/uploads/left", os.listdir("./turn_it_up/uploads/left")[0]), "r", encoding="UTF-8") as file:
                    left_file = re.split(r'(?<=[^A-Z].[.?!]) +(?=[A-Z])', file.read())[int(i)]
                with open(os.path.join("./turn_it_up/uploads/right", assignments[i]["document_name"] + ".txt"), "r", encoding="UTF-8") as file:
                    right_file = re.split(r'(?<=[^A-Z].[.?!]) +(?=[A-Z])', file.read())[int(assignments[i]["sentence_index"])]
                output += "<tr><td style = 'padding: 15px;'>" + str(int(i)+1) + "</td><td style = 'padding: 15px;'>" + left_file + "</td><td style = 'padding: 15px;'>" + assignments[i]["document_name"] + ".txt</td><td style = 'padding: 15px;'>" + str(int(assignments[i]["sentence_index"])+1) + "</td><td style = 'padding: 15px;'>" + right_file + "</td><td style = 'padding: 15px;'>" + str(assignments[i]["similarity"]) + "</td></tr>"
            output += "</table>"

            # point out removed frequent common words if there are any
            if os.path.exists("./turn_it_up/results/common_words.txt"):
                output += "<h3>Removed Common Words</h3>"
                with open("./turn_it_up/results/common_words.txt", "r", encoding="UTF-8") as file:
                    commons = file.read() 
                output += "<p>The common words that are removed from the documents while checking similarity are as follows:</p>"
                for i in commons.split("\n"):
                    output += "<p>" + i + "</p>"
            return ui.HTML(output)
        else:
            return ui.HTML("<h3>Please select 'Ready to Compare' to compare the documents.</h3>")
    
    @output
    @render.ui
    @reactive.Calc
    def similiar():
        output = ""
        if input.fileLeft() is None or input.filesRight() is None:
            return ui.HTML("<h3>Please complete the upload process first.</h3>")
        if input.compare():
            # read from assignments.json and overall_sim.json
            with open("./turn_it_up/results/assignments.json", "r", encoding="UTF-8") as file:
                assignments = json.load(file)
            with open("./turn_it_up/uploads/left/" + os.listdir("./turn_it_up/uploads/left")[0], "r", encoding="UTF-8") as file:
                left_file = file.read()

            # generate report
            left_file_split = re.split(r'(?<=[^A-Z].[.?!]) +(?=[A-Z])', left_file)

            # determine the color of the sentence according to the similarity for left file
            for i in assignments:
                color = None
                if assignments[i]["similarity"] > 0.8:
                    color = "red"
                elif assignments[i]["similarity"] > 0.6:
                    color = "orange"
                elif assignments[i]["similarity"] > 0.4:
                    color = "yellow"
                else:
                    color = "green"
                left_file_split[int(i)] = "<span style = 'background-color: " + color + ";'>" + left_file_split[int(i)] + "</span>"
                left_file = " ".join(left_file_split)
                
            with open("./turn_it_up/results/overall_sim.json", "r", encoding="UTF-8") as file:
                overall_sim = json.load(file)
            
            # compare similarity between left file and right files if there is any similarity
            for i in overall_sim:
                with open("./turn_it_up/uploads/right/" + i + ".txt", "r", encoding="UTF-8") as file:
                    right_file = file.read()
                right_file_split = re.split(r'(?<=[^A-Z].[.?!]) +(?=[A-Z])', right_file)
                for j in assignments:
                    if assignments[j]["document_name"] == i:
                        color = None
                        if assignments[j]["similarity"] > 0.8:
                            color = "red"
                        elif assignments[j]["similarity"] > 0.6:
                            color = "orange"
                        elif assignments[j]["similarity"] > 0.4:
                            color = "yellow"
                        else:
                            color = "green"
                        right_file_split[int(assignments[j]["sentence_index"])] = "<span style = 'background-color: " + color + ";'>" + right_file_split[int(assignments[j]["sentence_index"])] + "</span>"
                right_file = " ".join(right_file_split)

                output += "<h3 style = 'text-align: center;'>" + os.listdir("./turn_it_up/uploads/left")[0] + " and " + i + ".txt</h3>"
                output += "<div style='width: 100%;overflow:auto;'><div style='float:left; width: 50%; border-style: solid; border-width: 5px;border-radius: 5px;'>"
                output += "<p style = 'text-align: center;'>" + left_file + "</p>"
                output += "</div><div style='float:right; width: 50%;border-style: solid; border-width: 5px;border-radius: 5px;'>"
                output += "<p style = 'text-align: center;'>" + right_file + "</p>"
                output += "</div></div>"
            return ui.HTML(output)


# decapitalize all letters in the documents
def decapitalize(documents):
    for i in range(len(documents["left_files"])):
        for j in range(len(documents["left_files"][i]["content"])):
            documents["left_files"][i]["content"][j] = documents["left_files"][i]["content"][j].lower()
    for i in range(len(documents["right_files"])):
        for j in range(len(documents["right_files"][i]["content"])):
            documents["right_files"][i]["content"][j] = documents["right_files"][i]["content"][j].lower()
    return documents
# remove punctuations (done and checked)
def remove_punctuation(documents):
    for i in range(len(documents["left_files"])):
        for j in range(len(documents["left_files"][i]["content"])):
            documents["left_files"][i]["content"][j] = re.sub(r'[^\w\s]','',documents["left_files"][i]["content"][j])
    for i in range(len(documents["right_files"])):
        for j in range(len(documents["right_files"][i]["content"])):
            documents["right_files"][i]["content"][j] = re.sub(r'[^\w\s]','',documents["right_files"][i]["content"][j])
    return documents

# remove stopwords (done and checked)
def remove_stopwords(documents, stopword_count):
    # create vocabulary of left file sorted by frequency
    left_vocabulary = []
    for i in range(len(documents["left_files"])):
        left_vocabulary.append({})
        for j in range(len(documents["left_files"][i]["content"])):
            for word in documents["left_files"][i]["content"][j].split():
                if word in left_vocabulary[i]:
                    left_vocabulary[i][word] += 1
                else:
                    left_vocabulary[i][word] = 1
        left_vocabulary[i] = {k: v for k, v in sorted(left_vocabulary[i].items(), key=lambda item: item[1], reverse=True)}

    # create vocabulary of each right file sorted by frequency
    right_vocabulary = []
    for i in range(len(documents["right_files"])):
        right_vocabulary.append({})
        for j in range(len(documents["right_files"][i]["content"])):
            for word in documents["right_files"][i]["content"][j].split():
                if word in right_vocabulary[i]:
                    right_vocabulary[i][word] += 1
                else:
                    right_vocabulary[i][word] = 1
        right_vocabulary[i] = {k: v for k, v in sorted(right_vocabulary[i].items(), key=lambda item: item[1], reverse=True)}
    

    # find the most frequent common words in number of stopword_count between left file and each right file
    common_words = []
    for i in range(len(documents["right_files"])):
        common_words.append([])
        for word in left_vocabulary[0]:
            if word in right_vocabulary[i]:
                common_words[i].append(word)
                if len(common_words[i]) == stopword_count:
                    break

    
    # create left file-right file pairs removing stopwords
    for i in range(len(documents["left_files"])):
        for k in range(len(documents["left_files"][i]["content"])):
            words = (documents["left_files"][i]["content"][k]).split()
            for word in words:
                if word in common_words[i]:
                    words.remove(word)
            documents["left_files"][i]["content"][k] = " ".join(words)

    return documents, common_words


# check synonym
def check_synonym(documents):
    # Synonym check will be done by translating the sentence to another language and then translating it back to the original language

    # Turkish to English
    model_tr_en = "Helsinki-NLP/opus-mt-tr-en"
    tokenizer_tr_en = AutoTokenizer.from_pretrained(model_tr_en)
    model_tr_en = AutoModelForSeq2SeqLM.from_pretrained(model_tr_en)

    # English to Turkish
    model_en_tr = "Helsinki-NLP/opus-mt-tc-big-en-tr"
    tokenizer_en_tr = AutoTokenizer.from_pretrained(model_en_tr)
    model_en_tr = AutoModelForSeq2SeqLM.from_pretrained(model_en_tr)

    for i in range(len(documents["left_files"])):
        for j in range(len(documents["left_files"][i]["content"])):
            # Turkish to English
            tokenized_text = tokenizer_tr_en([documents["left_files"][i]["content"][j]], return_tensors = "pt")

            translation = model_tr_en.generate(**tokenized_text)
            translated_text = tokenizer_tr_en.batch_decode(translation, skip_special_tokens = True)

            # English to Turkish
            tokenized_text = tokenizer_en_tr([translated_text[0]], return_tensors = "pt")

            translation = model_en_tr.generate(**tokenized_text)
            translated_text = tokenizer_en_tr.batch_decode(translation, skip_special_tokens = True)

            documents["left_files"][i]["content"][j] = translated_text[0]
        
        for j in range(len(documents["right_files"][i]["content"])):
            # Turkish to English
            tokenized_text = tokenizer_tr_en([documents["right_files"][i]["content"][j]], return_tensors = "pt")

            translation = model_tr_en.generate(**tokenized_text)
            translated_text = tokenizer_tr_en.batch_decode(translation, skip_special_tokens = True)

            # English to Turkish
            tokenized_text = tokenizer_en_tr([translated_text[0]], return_tensors = "pt")

            translation = model_en_tr.generate(**tokenized_text)
            translated_text = tokenizer_en_tr.batch_decode(translation, skip_special_tokens = True)

            documents["right_files"][i]["content"][j] = translated_text[0]

    return documents

# lemmatization
def lemmatization(documents):

    with open('./turn_it_up/revisedDict.pkl', 'rb') as f:
        revisedDict = pickle.load(f)

    for i in range(len(documents["left_files"])):
        for j in range(len(documents["left_files"][i]["content"])):
            word_list = nltk.word_tokenize(documents["left_files"][i]["content"][j])
            for k in range(len(word_list)):
                word_list[k] = lemmatizer.findPos(word_list[k].lower(), revisedDict)[0][0][:-2]
            documents["left_files"][i]["content"][j] = " ".join(word_list)
        for j in range(len(documents["right_files"][i]["content"])):
            word_list = nltk.word_tokenize(documents["right_files"][i]["content"][j])
            for k in range(len(word_list)):
                word_list[k] = lemmatizer.findPos(word_list[k].lower(), revisedDict)[0][0][:-2]
            documents["right_files"][i]["content"][j] = " ".join(word_list)
    return documents

# stemming
def stemming(documents):
    turkStem=TurkishStemmer()
    for i in range(len(documents["left_files"])):
        for j in range(len(documents["left_files"][i]["content"])):
            word_list = nltk.word_tokenize(documents["left_files"][i]["content"][j])
            for k in range(len(word_list)):
                word_list[k] = turkStem.stemWord(word_list[k])
            documents["left_files"][i]["content"][j] = " ".join(word_list)
        for j in range(len(documents["right_files"][i]["content"])):
            word_list = nltk.word_tokenize(documents["right_files"][i]["content"][j])
            for k in range(len(word_list)):
                word_list[k] = turkStem.stemWord(word_list[k])
            documents["right_files"][i]["content"][j] = " ".join(word_list)
    return documents


# classification function:
def classify(docs):
    # create idx2label and label2idx for left and right files
    idx2label_left = {}
    idx2label_right = {}

    for i in range(len(docs["left_files"])):
        idx2label_left[i] = docs["left_files"][i]["name"]

    for i in range(len(docs["right_files"])):
        idx2label_right[i] = docs["right_files"][i]["name"]


    example_documents = []
    for i in range(len(docs["left_files"])):
        example_documents.append(docs["left_files"][i]["content"])

    will_compare = []
    for i in range(len(docs["right_files"])):
        will_compare.append(docs["right_files"][i]["content"])


    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings_left = []
    for document_l in example_documents:
        embeddings_left.append(model.encode(document_l, convert_to_tensor=True))

    embeddings_right = []
    for document_r in will_compare:
        embeddings_right.append(model.encode(document_r, convert_to_tensor=True))

    

    cosine_scores = []
    # Compute cosine-similarities
    for i in range(len(embeddings_left)):
        cosine_scores.append(util.pytorch_cos_sim(embeddings_left[i], embeddings_right[i]))
    
    
    similiars_dicts = {}

    for idx, j in enumerate(cosine_scores):
    # j is the index of the document in the will_compare (second digit)

        for idx2, i in enumerate(j):
        # i is the index of the sentence in the example document (first digit)
            for idx3, k in enumerate(i):
            # k is the index of the sentence in the document in the will_compare (third digit)
                if k > 0.5:
                    similiars_dicts[str(idx2)+str(idx)+str(idx3)] = float(str(k)[7:-1])
                else:
                    similiars_dicts[str(idx2)+str(idx)+str(idx3)] = 0

    # assign sentences to documents
    assigned_sentences = {}
    for i in range(len(example_documents[0])):
        vals = []
        sentences = []
        for j in range(len(will_compare)):
            a = []
            for pair in [ [key,val] for key,val in similiars_dicts.items() if key.startswith(str(i)+str(j))]:
                a.append(pair[1])
            most_similar = [ [key,val] for key,val in similiars_dicts.items() if key.startswith(str(i)+str(j))][a.index(max(a))][0]
            vals.append(similiars_dicts[most_similar])
            sentences.append(most_similar)
        # find the index of the max value
        max_index = vals.index(max(vals))
        if max(vals) > 0:
            assigned_sentences[i] = {"document_index":sentences[max_index][1],"sentence_index":sentences[max_index][2],"similarity":max(vals)}
        else:
            assigned_sentences[i] = ""


    # create a dictionary where document index in the assigned_sentences is changed to the name of the document
    assigned_sentences_names = {}
    for i in range(len(assigned_sentences)):
        if assigned_sentences[i] != "":
            assigned_sentences_names[i] = {"document_name":idx2label_right[int(assigned_sentences[i]["document_index"])],"sentence_index":assigned_sentences[i]["sentence_index"],"similarity":assigned_sentences[i]["similarity"]}


    ##############################################################################################
    #    This calculates the similarity of the sentences * the character length of the sentences #
    # for every similiar document, therefore a similarity ratio over the whole document will be  #
    # calculated.                                                                                #
    ##############################################################################################
    
    similarities_dict = {}
    for i in assigned_sentences:
        similarities_dict[assigned_sentences[i]['document_index']] = 0
    
    for i in assigned_sentences:
        if assigned_sentences[i]!= "":
            length_of_sentence = len(example_documents[int(assigned_sentences[i]['document_index'])][int(i)])
            length_of_document = len("".join(example_documents[int(assigned_sentences[i]['document_index'])]))
            similarities_dict[assigned_sentences[i]['document_index']] = similarities_dict[assigned_sentences[i]['document_index']] + ((length_of_sentence * assigned_sentences[i]['similarity'])/length_of_document)
                
    similarities = {}
    for i in similarities_dict:
        similarities[idx2label_right[int(i)]] = similarities_dict[i]


    return assigned_sentences_names, similarities


# delete the files remaining from the previous run
def empty_left():
    for f in os.listdir("./turn_it_up/uploads/left"):
        os.remove(os.path.join("./turn_it_up/uploads/left", f))

def empty_right():
    for f in os.listdir("./turn_it_up/uploads/right"):
        os.remove(os.path.join("./turn_it_up/uploads/right", f))

def empty_files():
    for f in os.listdir("./turn_it_up/results"):
        os.remove(os.path.join("./turn_it_up/results", f))


empty_left()
empty_right()
empty_files()

app = App(app_ui, server)
