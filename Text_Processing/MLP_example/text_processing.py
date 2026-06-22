# %% [markdown]
# # Multi Layer Perceptron

# %% [markdown]
# #Text Processing with 20 NewsGroup Dataset

# %%
#import package from scikit-learn containing 20newsgroups data
from sklearn.datasets import fetch_20newsgroups

full_train_data_labels = fetch_20newsgroups(subset = 'train') #to download the 20newsgroup dataset

#print the labels in the data
#print(type(full_train_data_labels.target_names)) #uncomment and check the output
print(list(set(full_train_data_labels.target_names)))

##Adding this for testing purpose, make this 0 for final run
test_epochs = 0


# %%
categories_list = ['comp.graphics', 'sci.med']

#Note that in the statement below, we fetch data concerned with the two categories in categories_list
train_data_labels = fetch_20newsgroups(subset='train',categories=categories_list, shuffle=True)


print(list(set(train_data_labels.target_names)))

#print a few documents and corresponding labels
num_documents = 3
for i in range(num_documents):
  print('Document:')
  print(train_data_labels.data[i])
  print('label:',train_data_labels.target[i])
  print('##########################################################')



# %%
from sklearn.feature_extraction.text import CountVectorizer

# %%
vectorizer = CountVectorizer()

# %%
#This is our simple dataset. It has four documents.
corpus = ['This is the first sample document.',
          'This is another document called the second sample document.',
          'And the third sample document.',
          'Is this the first sample document?',
        ]

# %%
X = vectorizer.fit_transform(corpus) #recall: vectorizer is initialized to countvectorizer.
print(type(X))
#print(X)

# %%
#unique words obtained from the corpus. Note: All words have lower case letters
print(vectorizer.get_feature_names_out())

# %% [markdown]
# $\textbf{Note:}$ CountVectorizer by default tokenizes string by extracting words of at least 2 letters.

# %%
print(X.toarray()) #Here is where we convert the text data into numerical data. Please understand this conversion process.

# %%
#This prints the vocabulary of unique tokens in the corpus with a corresponding index.
print(vectorizer.vocabulary_)

# %%
#to get the feature identifier of a particular word
print(vectorizer.vocabulary_.get('called'))

# %%
#we can use the vectorizer object created earlier to encode new sentences
#However if the new sentence contains new words, they would not be added to the vocabulary
vectorizer.transform(['completely new sentence.']).toarray()

# %%
vectorizer.transform(['another new sample document.']).toarray()

# %%
#if we want to rebuild the vectorizer we must include the new documents in corpus
#and then reconstruct again
#Consider this as a practice exercise and try adding new documents to the corpus and rebuild the vocabulary.

# Adding new documents to the corpus
corpus.append('This is a completely new document about data science created by Mamta.')
corpus.append('Another new document discussing machine learning created by MAmta.')

print("Updated Corpus:", corpus)

# Re-initializing and fitting the vectorizer on the updated corpus
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print('\nNew Vocabulary:')
print(vectorizer.get_feature_names_out())

print('\nTransformed Array (X.toarray()):')
print(X.toarray())

# %%
#recall: in our corpus the first document is 'This is the first sample document.',
#and the last document is 'Is this the first sample document?'
print(X.toarray()[0])
print(X.toarray()[-1])

# %% [markdown]
# Now we will use $\texttt{CountVectorizer()}$ to consider unigrams and bigrams.

# %%
#To account for the word order we can use bigrams and n-grams containing multiple words as features
#we will build another vectorizer using bigram features now
bigram_vectorizer = CountVectorizer(ngram_range=(1,2)) #consider both unigrams and bigrams

# %%
X_unigram_bigram = bigram_vectorizer.fit_transform(corpus)
print(type(X_unigram_bigram))

# %%
print(bigram_vectorizer.get_feature_names_out()) #watch for the bigrams along with unigrams now

# %%
print(len(bigram_vectorizer.get_feature_names_out()))

# %%
bigram_vectorizer.vocabulary_

# %%
#Now the feature vector is based on the dictionary above which has unigrams and bigrams.
X_unigram_bigram.toarray()

# %% [markdown]
# #Construction of X_unigram_bigram.

# %%
#Now let us compare the features of first and last document
print(X_unigram_bigram.toarray()[0])
print(X_unigram_bigram.toarray()[-1])

# %%
#Vectorizer with unigram, bigram and trigram features

# Create a CountVectorizer with unigram, bigram, and trigram features
trigram_vectorizer = CountVectorizer(ngram_range=(1,3))

# Fit and transform the corpus
X_unigram_bigram_trigram = trigram_vectorizer.fit_transform(corpus)

print(f"Shape of the feature matrix: {X_unigram_bigram_trigram.shape}")

# Print the new vocabulary (feature names)
print('\nVocabulary with unigrams, bigrams, and trigrams:')
# Display only a subset of feature names for readability due to potentially large output
feature_names = trigram_vectorizer.get_feature_names_out()
print(f"Total number of features: {len(feature_names)}")
print('\nfirst 20 features:')
print(feature_names[:20]) # Displaying first 20 features
print('...')
print('\nlast 20 features:')
print(feature_names[-20:]) # Displaying last 20 features

# Display the transformed array (first few rows for brevity)
print('\nTransformed Array (first 3 rows):')
print(X_unigram_bigram_trigram.toarray()[:3])

# %%
corpus_2 = [ 'The boy came to the school in the morning by the bus.',
             'He took his book and wrote in the book using his pencil.',
             'He played the game using the ball.',
             'He took his bag, left the school by the bus and reached his home.',
             ]

# %%
#a simple count based vectorizer gives too much significance to words like "the", "his"
vectorizer_corpus_2 = CountVectorizer()

# %%
X_corpus_2 = vectorizer_corpus_2.fit_transform(corpus_2)

# %%
print(vectorizer_corpus_2.get_feature_names_out())
print('num features:',len(vectorizer_corpus_2.get_feature_names_out()))

# %%
print(vectorizer_corpus_2.vocabulary_)

# %%
X_corpus_2.toarray()

# %% [markdown]
# Note that in $\texttt{X_corpus_2}$ we have large values for more frequent terms like $\texttt{the}$, $\texttt{his}$, etc.
# 
# Instead of a simple count based vectorizer, we will use a TF-IDF based feature vector construction.

# %%
#Let us transform the features using a tf-idf based weight scheme
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

# %%
tfidf_X_corpus_2 = tfidf_transformer.fit_transform(X_corpus_2) #Note: We are using X_corpus_2 to fit a TF-IDF transform.

# %%
print(tfidf_X_corpus_2.toarray())

# %% [markdown]
# Note that in $\texttt{tfidf_X_corpus_2}$ the weights of more frequent (appearing multiple times in most documents in the corpus) non-important vocabulary elements have been reduced.

# %%
#check if each row has unit Euclidean norm
import numpy as np
X1 = tfidf_X_corpus_2.toarray()
for i in range(len(X1)):
  print(np.linalg.norm(X1[i]))

# %%
# categories_list = ['comp.graphics', 'sci.med']

# #Note that in the statement below, we fetch data concerned with the two categories in categories_list
# train_data_labels = fetch_20newsgroups(subset='train',categories=categories_list, shuffle=True)

print('num documents:', len(train_data_labels.data))

# %%
data_20newsgroups_vectorizer_uni = CountVectorizer() #we consider only unigrams here

# %%
X_train_20newsgroups = data_20newsgroups_vectorizer_uni.fit_transform(train_data_labels.data)

# %%
features = data_20newsgroups_vectorizer_uni.get_feature_names_out()
num_features = len(features)
print('unigram features:', num_features)
print('unigram feature names:', features[:20]) #print first 20 feature names

# %%
print(X_train_20newsgroups.toarray().shape)

# %%
num_samples = X_train_20newsgroups.toarray().shape[0]
print('num samples:', num_samples)

# %%
y_train_20newsgroups = train_data_labels.target
print(y_train_20newsgroups.shape)

# %% [markdown]
# Shuffle and split $\texttt{(X_train_20newsgroups, y_train_20newsgroups)}$ into 80\% which you will use for training, and 20\% which you will use for validation.
# 
# Use specific names for these partitions.

# %%
from sklearn.model_selection import train_test_split

# Split the data into training and validation sets (80% train, 20% val)
X_train, X_val, y_train, y_val = train_test_split(X_train_20newsgroups, y_train_20newsgroups, test_size=0.2, random_state=42, shuffle=True)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_val:", y_val.shape)


# %% [markdown]
# Data loaders for loading the train and val data partitions.
# 

# %%
import torch
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 64

device = "cuda:0" if torch.cuda.is_available() else 'cpu' # For selecting the device, CPU or GPU.


#Convert the sparse matrices to dense tensors
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

#Wrap the tensors in a dataset and then create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

#Create data loaders for training and validation sets - 
# mini batches of data for training and validation
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Number of batches in train_loader:", len(train_loader))
print("Number of batches in val_loader:", len(val_loader))

first_text_batch, first_batch_labels = next(iter(train_loader)) #Gives 1st batch of the iterator
first_text_batch, first_batch_labels = first_text_batch.to(device), first_batch_labels.to(device)
first_text_batch.shape, first_batch_labels.shape

# %% [markdown]
# A simple MLP to classify the samples into the two categories.

# %%
#your network architecture and forward() function goes here
from torch import nn
from torch.nn import functional as F

print(BATCH_SIZE)

class MyMLP(nn.Module):
    def __init__(self, input_features, hidden_size, num_classes):
        super().__init__()

        # No need to Flatten the input as the input is already in the shape [batch, input_features]
        # self.flat = nn.Flatten()

        # print('num features:', input_features) #24614
        # ip -> h1, [batch, input_features] -> [batch, h1]
        self.layer1 = nn.Linear(in_features=input_features, out_features=hidden_size, bias=True) #500, 800, 1000

        # h1 -> h2, [batch, step_in] -> [batch, step_out]
        self.layer2 = nn.Linear(in_features=hidden_size, out_features=hidden_size//2, bias=True) #125, 250, 500

        # h2 -> output, [batch, 250] -> [batch, num_classes]  (2 categories: comp.graphics, sci.med)
        self.outer_layer = nn.Linear(in_features=hidden_size//2, out_features=num_classes, bias=True) #2 categories: comp.graphics, sci.med

    def forward(self, x):
        # x = self.flat(x)
        x = self.layer1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = F.relu(x)

        x = self.outer_layer(x)
        # No activation function here as CrossEntropyLoss expects raw logits

        return x
    
MODEL_NAME = 'MLP_20newsgroups'


# %%
from torchinfo import summary

#Create a model object

my_model = MyMLP(input_features=num_features, hidden_size=500, num_classes=2).to(device)
# my_model


summary(model=my_model,
        input_data=torch.randn(BATCH_SIZE, num_features).to(device), # Example input data with the correct shape
        col_names = ["input_size", "output_size", "num_params", "trainable", "params_percent"],
        col_width=20,
        row_settings=["var_names"],
        depth = 1,
        device=device
        )



# %%
#Set up a loss function
criterion_loss_fun = nn.CrossEntropyLoss()

#Set up an optimizer
optimizer = torch.optim.SGD(params=my_model.parameters(), lr=0.001)

logits = my_model(first_text_batch)
print(logits.shape)

logits[0,:]

loss_value = criterion_loss_fun(logits, first_batch_labels)
print(loss_value)

pred = logits.softmax(dim=1).argmax(dim=1)
print(pred)

# %%
#set model to train mode
#model.train()

#set number of epochs for training
#num_epochs = ??

#Write code for training your network using mini-batch optimization
import copy
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict

torch.manual_seed(42)
torch.cuda.manual_seed(42)
num_epochs = 50

if test_epochs>0:
    num_epochs = test_epochs

##Initialized here only for unigram and rereferneced throughout 
# the notebook for storing results in a common dictionary
comparision_dict_2class = defaultdict(dict)

best_cum_train_f1, best_cum_val_f1 = 0, 0
best_weights = None
best_epoch = np.inf
patience = 5

all_weight_norms = []
all_grad_norms = []

def results_initialize():
    return {"train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "best_cum_train_f1": [],
        "best_cum_val_f1": []
    }



my_model = MyMLP(input_features=num_features, hidden_size=500, num_classes=2).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = my_model.parameters(), lr = 0.001)

results = results_initialize()

start_time = timer()
for epoch in range(num_epochs):

    weight_norms = []
    grad_norms = []


    ### Training loop
    cum_train_loss, cum_train_acc, cum_train_f1 = 0, 0, 0

    my_model.train() # Sets the model to training mode, enabling features like dropout and batch normalization
    for batch, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)

        logits = my_model(x_train)

        loss = loss_function(logits, y_train)
        cum_train_loss += loss.detach().cpu().numpy()


        optimizer.zero_grad() # reset the gradients

        loss.backward() # compute the gradients of the loss with respect to the model parameters

        optimizer.step() # updates the model's parameters based on the gradients

        pred = logits.softmax(dim=1).argmax(dim=1)

        acc = 100*accuracy_score(y_train.detach().cpu().numpy(), pred.detach().cpu().numpy())
        cum_train_acc += acc

        f1 = f1_score(y_train.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='weighted')
        cum_train_f1 += f1

    ### Book keeping of weight norms and grad norms
    for layer in my_model.children():
        if isinstance(layer, torch.nn.Linear):
            weight_norms.append(layer.weight.norm(2).item())
            grad_norms.append(layer.weight.grad.norm(2).item())


    cum_train_loss /= len(train_loader)
    cum_train_acc /= len(train_loader)
    cum_train_f1 /= len(train_loader)

    if cum_train_f1 > best_cum_train_f1:
        best_cum_train_f1 = cum_train_f1

    ### validation
    cum_val_loss, cum_val_acc, cum_val_f1 = 0, 0, 0

    my_model.eval() # Sets the model to evaluation mode, disabling features like dropout and batch normalization
    with torch.inference_mode(): # Disables gradient computation, optimizing performance during inference
        for batch, (x_val, y_val) in enumerate(val_loader):
            x_val, y_val = x_val.to(device), y_val.to(device)

            logits = my_model(x_val)

            loss = loss_function(logits, y_val)
            cum_val_loss += loss.detach().cpu().numpy()

            pred = logits.softmax(dim=1).argmax(dim=1)

            acc = 100*accuracy_score(y_val.detach().cpu().numpy(), pred.detach().cpu().numpy())
            cum_val_acc += acc

            f1 = f1_score(y_val.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='weighted')
            cum_val_f1 += f1

    cum_val_loss /= len(val_loader)
    cum_val_acc /= len(val_loader)
    cum_val_f1 /= len(val_loader)


    if cum_val_f1 > best_cum_val_f1:
        best_cum_val_f1 = cum_val_f1
        best_weights = copy.deepcopy(my_model.state_dict())
        best_epoch = epoch
        patience = 5
    else:
        ### Early Stoping
        patience -= 1
        if patience == 0:
            break

    # Adding NA for report comparision, can calculate these later
    results["train_loss"].append(cum_train_loss)
    results["val_loss"].append(cum_val_loss)
    results["train_acc"].append(cum_train_acc)
    results["val_acc"].append(cum_val_acc)
    results["train_f1"].append(cum_train_f1)
    results["val_f1"].append(cum_val_f1)
    results["train_precision"].append(float('nan'))
    results["train_recall"].append(float('nan'))
    results["val_precision"].append(float('nan'))
    results["val_recall"].append(float('nan'))


    if epoch % 1 == 0:
        print(f"Epoch: {epoch} | "
              f"Best_cum_F1: ({best_cum_train_f1:.4f}, {best_cum_val_f1:.4f}) | "
              f"Loss: ({cum_train_loss:.4f}, {cum_val_loss:.4f}) | "
              f"Acc: ({cum_train_acc:.4f}%, {cum_val_acc:.4f}%) | "
              f"F1: ({cum_train_f1:.4f}, {cum_val_f1:.4f}) |")
        results["best_cum_val_f1"].append(best_cum_val_f1)
        results["best_cum_train_f1"].append(best_cum_train_f1)

    all_weight_norms.append(weight_norms)
    all_grad_norms.append(grad_norms)

comparision_dict_2class[my_model.__class__.__name__]['2ClassUnigram'] = copy.deepcopy(results)




# %% [markdown]
# Plot the loss and accuracy values obtained on train and val split.

# %%
import matplotlib.pyplot as plt
import torch

plt.plot(np.arange(1, 1+len(results['train_loss'])), results["train_loss"], label='Train Loss', color='blue')
plt.plot(np.arange(1, 1+len(results['val_loss'])), results["val_loss"], label='val Loss', color='orange')
plt.ylabel('loss')

# plt.subplot(10, 5, 20)
# for i in range(len(all_weight_norms[0])):
#     weight_norms = [epoch_norms[i] for epoch_norms in all_weight_norms]
#     plt.plot(range(1, len(weight_norms) + 1), weight_norms, label=f'Layer {i+1} Weight Norm')
# plt.title('L2 Norm of Weights vs Epochs')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# %%
import os
### Saving Best Model weights
os.makedirs('checkpoints', exist_ok=True)
torch.save(best_weights, f"checkpoints/{MODEL_NAME}_checkpoint_{best_epoch}.pth.tar")
print(f'Model checkpoint saved at epoch {best_epoch}')

end_time = timer()
print(f"Train time is ", end_time-start_time, f"Best val F1: {best_cum_val_f1:.4f}")

plt.figure(figsize=(10, 17))

plt.subplot(6, 1, 1)
plt.plot(np.arange(1, 1+len(results['train_loss'])), results["train_loss"], label='Train Loss', color='blue')
plt.plot(np.arange(1, 1+len(results['val_loss'])), results["val_loss"], label='val Loss', color='orange')
plt.title('Loss')
plt.legend()

plt.subplot(6, 1, 2)
plt.plot(np.arange(1, 1+len(results['train_acc'])), results["train_acc"], label='Train Accuracy', color='blue')
plt.plot(np.arange(1, 1+len(results['val_acc'])), results["val_acc"], label='val Accuracy', color='orange')
plt.title('Accuracy')
plt.legend()

plt.subplot(6, 1, 3)
plt.plot(np.arange(1, 1+len(results['train_f1'])), results["train_f1"], label='Train F1 Score', color='blue')
plt.plot(np.arange(1, 1+len(results['val_f1'])), results["val_f1"], label='val F1 Score', color='orange')
plt.title('F1 Score')
plt.legend()

plt.subplot(6, 1, 4)
plt.plot(np.arange(1, 1+len(results['best_cum_train_f1'])), results["best_cum_train_f1"], label='Best Train F1 Score', color='blue')
plt.plot(np.arange(1, 1+len(results['best_cum_val_f1'])), results["best_cum_val_f1"], label='Best val F1 Score', color='orange')
plt.title('Best F1 Score')
plt.legend()

plt.subplot(6, 1, 5)
for i in range(len(all_weight_norms[0])):
    weight_norms = [epoch_norms[i] for epoch_norms in all_weight_norms]
    plt.plot(range(1, len(weight_norms) + 1), weight_norms, label=f'Layer {i+1} Weight Norm')
plt.title('L2 Norm of Weights vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('L2 Norm of Weights')
plt.legend()

plt.subplot(6, 1, 6)
for i in range(len(all_grad_norms[0])):
    grad_norms = [epoch_norms[i] for epoch_norms in all_grad_norms]
    plt.plot(range(1, len(grad_norms) + 1), grad_norms, label=f'Layer {i+1} Gradient Norm')
plt.title('L2 Norm of Gradients vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('L2 Norm of Gradients')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# #Evaluate your model on test data, that is already provided in the original data source.

# %%
test_data_labels = fetch_20newsgroups(subset='test',categories=categories_list, shuffle=True)

X_test_20newsgroups = data_20newsgroups_vectorizer_uni.transform(test_data_labels.data) # we only use transform
print(X_test_20newsgroups.shape)
y_test_20newsgroups = test_data_labels.target

num_test_samples = X_test_20newsgroups.toarray().shape[0]
print('num test samples:',num_test_samples)



# %% [markdown]
# Set up a data loader for the test data.

# %%
#Data loader for test
#Create data loaders for test set

#Convert the sparse matrices to dense tensors
X_test_tensor = torch.tensor(X_test_20newsgroups.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_20newsgroups, dtype=torch.long)

#Wrap the tensors in a dataset and then create data loaders
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create data loaders for training and validation sets
# For the test (or validation) set, shuffle=False is used to keep the data order fixed. 
# This ensures reproducibility and consistent evaluation, as we want to measure 
# performance on the exact same data in the same order every time.
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Number of batches in test_loader:", len(test_loader))



# %% [markdown]
# Use the trained model to evaluate accuracy, F1, Precision, Recall on the Test Data

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#set the neural net model to eval mode for testing
my_model.eval() # Sets the model to evaluation mode, disabling features like dropout and batch normalization

#Write code for evaluation
### validation
cum_test_loss, cum_test_acc, cum_test_f1, cum_test_precision, cum_test_recall = 0, 0, 0, 0, 0
with torch.inference_mode(): # Disables gradient computation, optimizing performance during inference
    for batch, (x_val, y_val) in enumerate(test_loader):
        x_val, y_val = x_val.to(device), y_val.to(device)

        logits = my_model(x_val)

        loss = loss_function(logits, y_val)
        cum_test_loss += loss.detach().cpu().numpy()

        pred = logits.softmax(dim=1).argmax(dim=1)

        acc = 100*accuracy_score(y_val.detach().cpu().numpy(), pred.detach().cpu().numpy())
        cum_test_acc += acc

        f1 = f1_score(y_val.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='weighted')
        cum_test_f1 += f1

        precision = precision_score(y_val.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='weighted', zero_division=0)
        cum_test_precision += precision

        recall = recall_score(y_val.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='weighted')
        cum_test_recall += recall


print("Accuracy", cum_test_acc / len(test_loader))
print("F1 Score", cum_test_f1 / len(test_loader))
print("Precision", cum_test_precision / len(test_loader))
print("Recall", cum_test_recall / len(test_loader))



# %% [markdown]
# # Evaluate the model on another data corpus.

# %%
newsgroup_test_corpus = ['New The monitor in operation theatre has some snag',
                         'New What is the RAM size used in the PC?',
                         'New RBC and Haemoglobin are below normal.',
                         'New Renal disease is on the rise.',
                         'Some New random Text about graphics and medical science.',
                         'New Neurologists believe that this case is rare.']
X_test_corpus = data_20newsgroups_vectorizer_uni.transform(newsgroup_test_corpus) # we only use transform
print(X_test_corpus.shape)

num_test_samples = X_test_corpus.toarray().shape[0]
print('num test samples:',num_test_samples)

my_model.eval()

x_test = torch.FloatTensor(X_test_corpus.toarray())

y_test_pred = my_model(x_test)

# y_test_pred_cpu = y_test_pred.squeeze().detach().cpu().numpy()
y_test_pred_cpu = y_test_pred.softmax(dim=1).argmax(dim=1).detach().cpu().numpy()
y_test_pred_cpu = np.multiply(y_test_pred_cpu>0.5,1)

print('predictions:')
print(y_test_pred_cpu)

y_true = [0, 0, 1, 1, 1, 1] 
accuracy = accuracy_score(y_true, y_test_pred_cpu)
print('Accuracy:', accuracy)


# %%
#Unigram and bigram features together
from sklearn.metrics import confusion_matrix


RANDOM_STATE = 42
device = "cuda:0" if torch.cuda.is_available() else 'cpu' # For selecting the device, CPU or GPU.

##Creating methods for data preprocessing, 
# training and evaluation to avoid code repetition 
# And many mistakes

def data_preprocess(categories_list, range_tuple, max_features=None, feature_type=None):
    # Step 1: Fetch the data for the specified categories
    train_data_labels = fetch_20newsgroups(subset='train', categories=categories_list)
    corpus = train_data_labels.data                # list of documents (raw text)
    y_train_current = train_data_labels.target  # array of labels for each document

    # Split the data into training and validation sets
    X_train_set, X_val_set, y_train_set, y_val_set = train_test_split(corpus, y_train_current, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)

    # Step 2: Vectorize: Fit on train - Create a CountVectorizer with unigram and bigram features
    data_20newsgroups_vectorizer = CountVectorizer(ngram_range=range_tuple, max_features=max_features) #consider both unigrams and bigrams
    x_train = data_20newsgroups_vectorizer.fit_transform(X_train_set)
    x_val = data_20newsgroups_vectorizer.transform(X_val_set)

    feature_names = data_20newsgroups_vectorizer.get_feature_names_out()
    num_features = len(feature_names)
    print(f'{feature_type} features:')
    print('num features:', num_features)
    print('feature names:', feature_names[:20]) #print first 20 feature names
    print("Training features shape:", x_train.shape)
    print("Validation features shape:", x_val.shape)

    ## Step 3: Using TFID Transformer to transform 
    # the count based features to TF-IDF based features
    # Fit on train, transform on val
    tfidf_transformer = TfidfTransformer()
    X_train_tfid = tfidf_transformer.fit_transform(x_train)
    X_val_tfid = tfidf_transformer.transform(x_val)

    print("Transformed Training features shape:", X_train_tfid.shape)
    print(X_train_tfid.toarray())

    #Step 4: Convert the sparse matrices to dense tensors 
    #Convert the sparse matrices to dense tensors
    X_train_tensor = torch.tensor(X_train_tfid.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_set, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_tfid.toarray(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_set, dtype=torch.long)

    #Step 5: Mini-batch - Data loader for train and val
    #Wrap the tensors in a dataset and then create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return num_features, train_loader, val_loader, data_20newsgroups_vectorizer

def train_model(model, train_loader, loss_function, optimizer):
    
    model.train()
    cum_loss, cum_acc, cum_f1, cum_precision, cum_recall = 0, 0, 0, 0, 0
    for batch, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)
        logits = model(x_train)
        loss = loss_function(logits, y_train)
        cum_loss += loss.detach().cpu().numpy()
        optimizer.zero_grad() # reset the gradients
        loss.backward() # compute the gradients of the loss with respect to the model parameters
        optimizer.step() # updates the model's parameters based on the gradients

        pred = logits.softmax(dim=1).argmax(dim=1)
        
        acc = 100*accuracy_score(y_train.detach().cpu().numpy(), pred.detach().cpu().numpy())
        cum_acc += acc
        
        f1 = f1_score(y_train.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='weighted')
        cum_f1 += f1

        precision = precision_score(y_train.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='weighted', zero_division=0)
        cum_precision += precision

        recall = recall_score(y_train.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='weighted')
        cum_recall += recall

    return cum_loss / len(train_loader), cum_acc / len(train_loader), cum_f1 / len(train_loader), cum_precision / len(train_loader), cum_recall / len(train_loader)


# Evaluation function
def evaluate_model(model, loader, loss_function):
    model.eval() # Sets the model to evaluation mode, disabling features like dropout and batch normalization
    cum_loss, cum_acc, cum_f1, cum_precision, cum_recall = 0, 0, 0, 0, 0
    val_pred = []
    val_labels = []
    with torch.inference_mode(): # Disables gradient computation, optimizing performance during inference
        for batch, (x_val, y_val) in enumerate(loader):
            x_val, y_val = x_val.to(device), y_val.to(device)
            logits = model(x_val)
            loss = loss_function(logits, y_val)
            cum_loss += loss.detach().cpu().numpy()
            pred = logits.softmax(dim=1).argmax(dim=1)

            val_pred.extend(pred.detach().cpu().numpy())
            val_labels.extend(y_val.detach().cpu().numpy())
            
            acc = 100*accuracy_score(y_val.detach().cpu().numpy(), pred.detach().cpu().numpy())
            cum_acc += acc
            
            f1 = f1_score(y_val.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='weighted')
            cum_f1 += f1
            precision = precision_score(y_val.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='weighted', zero_division=0)
            cum_precision += precision

            recall = recall_score(y_val.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='weighted')
            cum_recall += recall

    cm = confusion_matrix(val_labels, val_pred)
    print("Confusion Matrix:\n", cm)

    return cum_loss / len(loader), cum_acc / len(loader), cum_f1 / len(loader), cum_precision / len(loader), cum_recall / len(loader)


def epoch_wise_training(model, train_loader, val_loader, loss_function, optimizer, num_epochs, cat_type, comparision_dict):

    if test_epochs>0:
        num_epochs = test_epochs

    best_cum_train_f1, best_cum_val_f1 = 0, 0
    best_weights = None
    best_epoch = np.inf
    patience = 5

    results = results_initialize()

    # all_weight_norms = []
    # all_grad_norms = []

    # start_time = timer()
    for epoch in range(num_epochs):

        weight_norms = []
        grad_norms = []

        # 1. Train the model for one epoch
        cum_train_loss, cum_train_acc, cum_train_f1, cum_train_precision, cum_train_recall = 0, 0, 0, 0, 0
        (cum_train_loss, cum_train_acc, cum_train_f1, cum_train_precision, cum_train_recall) = train_model(model, train_loader, loss_function, optimizer)
        print(f"Training Loss: {cum_train_loss:.4f}, Accuracy: {cum_train_acc:.2f}%, F1 Score: {cum_train_f1:.4f}, Precision: {cum_train_precision:.4f}, Recall: {cum_train_recall:.4f}")

        ### Book keeping of weight norms and grad norms
        for layer in model.children():
            if isinstance(layer, torch.nn.Linear):
                weight_norms.append(layer.weight.norm(2).item())
                grad_norms.append(layer.weight.grad.norm(2).item())
        
        if cum_train_f1 > best_cum_train_f1:
            best_cum_train_f1 = cum_train_f1

        #Evaluate Validation - val_ub_loader
        cum_val_loss, cum_val_acc, cum_val_f1, cum_val_precision, cum_val_recall = 0, 0, 0, 0, 0
        (cum_val_loss, cum_val_acc, cum_val_f1, cum_val_precision, cum_val_recall) = evaluate_model(model, val_loader, loss_function)
        print(f"Validation Loss: {cum_val_loss:.4f}, Accuracy: {cum_val_acc:.2f}%, F1 Score: {cum_val_f1:.4f}, Precision: {cum_val_precision:.4f}, Recall: {cum_val_recall:.4f}")

        if cum_val_f1 > best_cum_val_f1:
            best_cum_val_f1 = cum_val_f1
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 5
        else:
            ### Early Stopping
            patience -= 1
            if patience == 0:
                break

        results["train_loss"].append(cum_train_loss)
        results["val_loss"].append(cum_val_loss)
        results["train_acc"].append(cum_train_acc)
        results["val_acc"].append(cum_val_acc)
        results["train_f1"].append(cum_train_f1)
        results["val_f1"].append(cum_val_f1)
        results["train_precision"].append(cum_train_precision)
        results["val_precision"].append(cum_val_precision)
        results["train_recall"].append(cum_train_recall)
        results["val_recall"].append(cum_val_recall)
        results["best_cum_train_f1"].append(best_cum_train_f1)
        results["best_cum_val_f1"].append(best_cum_val_f1)


        if epoch % 1 == 0:
            print(f"Epoch: {epoch} | "
                f"Best_cum_F1: ({best_cum_train_f1:.4f}, {best_cum_val_f1:.4f}) | "
                f"Loss: ({cum_train_loss:.4f}, {cum_val_loss:.4f}) | "
                f"Acc: ({cum_train_acc:.4f}%, {cum_val_acc:.4f}%) | "
                f"F1: ({cum_train_f1:.4f}, {cum_val_f1:.4f}) | "
                f"Precision: ({cum_train_precision:.4f}, {cum_val_precision:.4f}) | "
                f"Recall: ({cum_train_recall:.4f}, {cum_val_recall:.4f}) |")
            
    comparision_dict[model.__class__.__name__][cat_type] = copy.deepcopy(results)


    return comparision_dict

def test_data_evaluation(data_20newsgroups_vectorizer, categories_list, model, criterion):
    # Test set
    test_data_labels = fetch_20newsgroups(subset='test', categories=categories_list, shuffle=True)
    X_test = data_20newsgroups_vectorizer.transform(test_data_labels.data)
    y_test = test_data_labels.target
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False)

    #Evaluate Test - test_loader
    cum_test_loss, cum_test_acc, cum_test_f1, cum_test_precision, cum_test_recall = 0, 0, 0, 0, 0
    (cum_test_loss, cum_test_acc, cum_test_f1, cum_test_precision, cum_test_recall) = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {cum_test_loss:.4f}, Accuracy: {cum_test_acc:.2f}%, F1 Score: {cum_test_f1:.4f}, Precision: {cum_test_precision:.4f}, Recall: {cum_test_recall:.4f}")

import pandas as pd
def compare_models(comparision_dict):

    print("=" * 120)
    print("COMPARISON REPORT")
    print("=" * 120)

    for model_name, feature_dict in comparision_dict.items():
        # all epocs aren't same due to early stop - patience 5, 
        # so getting minimu epochs across all features 
        # for a model to report comparision
        min_epochs = min(len(metrics['train_loss']) for metrics in feature_dict.values())

        #For each Model epoch print category/feature and metrics
        for i in range(min_epochs):
            for feature_name, metrics in feature_dict.items():        
                print(
                    f"{model_name:<5} | {feature_name:<15} | {i:2d} | "
                    f"Train Loss: {metrics['train_loss'][i]:.2f}, "
                    f"Val Loss: {metrics['val_loss'][i]:.2f}, "
                    f"Train Accuracy: {metrics['train_acc'][i]:.2f}, "
                    f"Val Accuracy: {metrics['val_acc'][i]:.2f}, "
                    f"Train F1: {metrics['train_f1'][i]:.2f}, "
                    f"Val F1: {metrics['val_f1'][i]:.2f}, "
                    f"Train Precision: {metrics['train_precision'][i]:.2f}, "
                    f"Val Precision: {metrics['val_precision'][i]:.2f}, "
                    f"Train Recall: {metrics['train_recall'][i]:.2f}, "
                    f"Val Recall: {metrics['val_recall'][i]:.2f}"
                )

    print("=" * 120)


# %%
########

# 1a. Use unigram and bigram features and perform training using an appropriate feed forward network 
# (or) MLP. Compute train set, val set and test set accuracy, F1 score, Precision, 
# Recall and Confusion matrix.

#Limiting max features to 25,000 to avoid memory issues. 
num_features_ub, train_ub_loader, val_ub_loader, data_20newsgroups_vectorizer_uni_bigram = data_preprocess(categories_list, range_tuple=(1,2), max_features=25000, feature_type='2ClassUniBigram')

my_model_ub = MyMLP(input_features=num_features_ub, hidden_size=5000, num_classes=len(categories_list)).to(device)

summary(model=my_model_ub,
        input_data=torch.randn(BATCH_SIZE, num_features_ub).to(device), # Example input data with the correct shape
        col_names = ["input_size", "output_size", "num_params", "trainable", "params_percent"],
        col_width=20,
        row_settings=["var_names"],
        depth = 1,
        device=device
        )


# %%
##Training the model with unigram and bigram features together

criterion_ub = torch.nn.CrossEntropyLoss()
##changing learning rate to 0.0005 for the unigram+bigram model as 
# it has more parameters and is more complex than the unigram only model
## I can use Adam optimizer as well
optimizer_ub = torch.optim.SGD(my_model_ub.parameters(), lr=0.001)

comparision_dict_2class = epoch_wise_training(my_model_ub, train_ub_loader, val_ub_loader, criterion_ub, optimizer_ub, num_epochs=50, cat_type='2ClassUniBigram', comparision_dict=comparision_dict_2class)



# %%
test_data_evaluation(data_20newsgroups_vectorizer_uni_bigram, categories_list, my_model_ub, criterion_ub)

# # Test set
# test_data_labels = fetch_20newsgroups(subset='test', categories=categories_list, shuffle=True)
# X_test_ub = data_20newsgroups_vectorizer_uni_bigram.transform(test_data_labels.data)
# y_test_ub = test_data_labels.target
# X_test_tensor_ub = torch.tensor(X_test_ub.toarray(), dtype=torch.float32)
# y_test_tensor_ub = torch.tensor(y_test_ub, dtype=torch.long)
# test_loader_ub = DataLoader(TensorDataset(X_test_tensor_ub, y_test_tensor_ub), batch_size=BATCH_SIZE, shuffle=False)

# #Evaluate Test - test_loader_ub
# cum_test_loss, cum_test_acc, cum_test_f1, cum_test_precision, cum_test_recall = 0, 0, 0, 0, 0
# (cum_test_loss, cum_test_acc, cum_test_f1, cum_test_precision, cum_test_recall) = evaluate_model(my_model_ub, test_loader_ub, criterion_ub)
# print(f"Test Loss: {cum_test_loss:.4f}, Accuracy: {cum_test_acc:.2f}%, F1 Score: {cum_test_f1:.4f}, Precision: {cum_test_precision:.4f}, Recall: {cum_test_recall:.4f}")

# %%
compare_models(comparision_dict_2class)

# %% [markdown]
# 1. (b) Use unigram, bigram and trigram features in the code above and perform training using an appropriate feed forward network. Compute train set, val set and test set accuracy, F1 score, Precision, Recall and Confusion matrix.
# 
# 

# %%
########

num_features_ubt, train_ubt_loader, val_ubt_loader, data_20newsgroups_vectorizer_uni_bi_trigram = data_preprocess(categories_list, range_tuple=(1,3), max_features=30000, feature_type='2ClassUniBiTri')

my_model_ubt = MyMLP(input_features=num_features_ubt, hidden_size=5000, num_classes=len(categories_list)).to(device)

summary(model=my_model_ubt,
        input_data=torch.randn(BATCH_SIZE, num_features_ubt).to(device), # Example input data with the correct shape
        col_names = ["input_size", "output_size", "num_params", "trainable", "params_percent"],
        col_width=20,
        row_settings=["var_names"],
        depth = 1,
        device=device
        )

# %%
##Training the model with unigram, bigram, and trigram features together

criterion_ubt = torch.nn.CrossEntropyLoss()
##changing learning rate to 0.0005 for the unigram+bigram+trigram model as 
# it has more parameters and is more complex than the unigram only model
optimizer_ubt = torch.optim.SGD(my_model_ubt.parameters(), lr=0.0005)

comparision_dict_2class = epoch_wise_training(my_model_ubt, train_ubt_loader, val_ubt_loader, criterion_ubt, optimizer_ubt, num_epochs=50, cat_type='2ClassUniBiTri', comparision_dict=comparision_dict_2class)


# %%

test_data_evaluation(data_20newsgroups_vectorizer_uni_bi_trigram, categories_list, my_model_ubt, criterion_ubt)


# Test set for unigram+bigram+trigram model
# test_data_labels = fetch_20newsgroups(subset='test', categories=categories_list, shuffle=True)
# X_test_ubt = data_20newsgroups_vectorizer_uni_bi_trigram.transform(test_data_labels.data)
# y_test_ubt = test_data_labels.target
# X_test_tensor_ubt = torch.tensor(X_test_ubt.toarray(), dtype=torch.float32)
# y_test_tensor_ubt = torch.tensor(y_test_ubt, dtype=torch.long)
# test_loader_ubt = DataLoader(TensorDataset(X_test_tensor_ubt, y_test_tensor_ubt), batch_size=BATCH_SIZE, shuffle=False)

# #Evaluate Test - test_loader_ubt
# cum_test_loss, cum_test_acc, cum_test_f1, cum_test_precision, cum_test_recall = 0, 0, 0, 0, 0
# (cum_test_loss, cum_test_acc, cum_test_f1, cum_test_precision, cum_test_recall) = evaluate_model(my_model_ubt, test_loader_ubt, criterion_ubt)
# print(f"Test Loss: {cum_test_loss:.4f}, Accuracy: {cum_test_acc:.2f}%, F1 Score: {cum_test_f1:.4f}, Precision: {cum_test_precision:.4f}, Recall: {cum_test_recall:.4f}")

# %% [markdown]
# 1. (c) Compare and contrast the train set and test set performance metrics for parts (a) and (b) with those obtained in the code above where we used only unigrams. Discuss.

# %%
## To Discuss with Professor / TA
## Initially I wasn't seeing much difference in acc, f1, precision and recall scores 
#  across the uni+bi and uni+bi+tri models, 
## after so much trial, I am seeing some significant difference 
# in the scores across the models, but the scores are still low.   
# 
compare_models(comparision_dict_2class) 

# %% [markdown]
# 2. Using $k$ favorite classes of newsgroups and perform multi-class classification training using the feed forward network (by modifying the output layer). Use unigram, bigram and trigram features. Compute train set, val set and test set accuracy, F1 score, Precision, Recall and Confusion matrix.
# 

# %%
# Select k classes:
# Choose k (e.g., 3, 5, or any number you like) categories from the 20 newsgroups dataset to focus on, instead of using all 20.
# ['rec.sport.hockey', 'talk.religion.misc', 'comp.sys.mac.hardware', 'talk.politics.guns'
# , 'sci.crypt', 'sci.med', 'rec.autos', 'talk.politics.mideast', 'sci.space'
# , 'soc.religion.christian', 'comp.sys.ibm.pc.hardware', 'rec.sport.baseball'
# , 'sci.electronics', 'rec.motorcycles', 'alt.atheism', 'talk.politics.misc'
# , 'comp.graphics', 'comp.windows.x', 'misc.forsale', 'comp.os.ms-windows.misc']
K_classes_cat = ['sci.crypt', 'talk.politics.mideast', 'sci.space']

## Referenced here only and used thoughout the code for 
# k class classification
comparision_dict_k_class = defaultdict(dict)

# Multi-class classification:
# Train a model to distinguish between these k classes (not just binary classification).
# Modify the output layer:
# Change your neural network’s final layer to have k output units (one for each selected class), 
# so it can predict which class each sample belongs to.

## Unigram features for k classes
num_features_k_class_uni, train_k_class_loader_uni, val_k_class_loader_uni, data_20newsgroups_k_class_vectorizer_uni = data_preprocess(K_classes_cat, range_tuple=(1,1), max_features=20000, feature_type='KClassUnigram')

k_class_model_uni = MyMLP(input_features=num_features_k_class_uni, hidden_size=500, num_classes=len(K_classes_cat)).to(device)

summary(model=k_class_model_uni,
        input_data=torch.randn(BATCH_SIZE, num_features_k_class_uni).to(device), # Example input data with the correct shape
        col_names = ["input_size", "output_size", "num_params", "trainable", "params_percent"],
        col_width=20,
        row_settings=["var_names"],
        depth = 1,
        device=device
        )

# %%
##Training the model for K Classes with unigram

k_class_criterion_uni = torch.nn.CrossEntropyLoss()
##changing learning rate to 0.001 for the unigram model
k_class_optimizer_uni = torch.optim.SGD(k_class_model_uni.parameters(), lr=0.001)

comparision_dict_k_class = epoch_wise_training(k_class_model_uni, train_k_class_loader_uni, val_k_class_loader_uni, k_class_criterion_uni, k_class_optimizer_uni, num_epochs=50, cat_type='KClassUnigram', comparision_dict=comparision_dict_k_class)


# %%
##Test Data Evaluation for K classes with unigram features

test_data_evaluation(data_20newsgroups_k_class_vectorizer_uni, K_classes_cat, k_class_model_uni, k_class_criterion_uni)



# %%
##K Class model with unigram and bigram features

num_features_k_class_uni_bi, train_k_class_loader_uni_bi, val_k_class_loader_uni_bi, data_20newsgroups_k_class_vectorizer_uni_bi = data_preprocess(K_classes_cat, range_tuple=(1,2), max_features=25000, feature_type='KClassUniBigram')

k_class_model_uni_bi = MyMLP(input_features=num_features_k_class_uni_bi, hidden_size=5000, num_classes=len(K_classes_cat)).to(device)

summary(model=k_class_model_uni_bi,
        input_data=torch.randn(BATCH_SIZE, num_features_k_class_uni_bi).to(device), # Example input data with the correct shape
        col_names = ["input_size", "output_size", "num_params", "trainable", "params_percent"],
        col_width=20,
        row_settings=["var_names"],
        depth = 1,
        device=device
        )

##Training the model for K Classes with unigram and bigram features

k_class_criterion_uni_bi = torch.nn.CrossEntropyLoss()
##changing learning rate to 0.001 for the unigram model
k_class_optimizer_uni_bi = torch.optim.SGD(k_class_model_uni_bi.parameters(), lr=0.001)

comparision_dict_k_class = epoch_wise_training(k_class_model_uni_bi, train_k_class_loader_uni_bi, val_k_class_loader_uni_bi, k_class_criterion_uni_bi, k_class_optimizer_uni_bi, num_epochs=50, cat_type='KClassUniBigram', comparision_dict=comparision_dict_k_class)


##Test Data Evaluation for K classes with unigram and bigram features

test_data_evaluation(data_20newsgroups_k_class_vectorizer_uni_bi, K_classes_cat, k_class_model_uni_bi, k_class_criterion_uni_bi)




# %%
## K-Class model with unigram, bigram, and trigram features

num_features_k_class_uni_bi_tri, train_k_class_loader_uni_bi_tri, val_k_class_loader_uni_bi_tri, data_20newsgroups_k_class_vectorizer_uni_bi_tri = data_preprocess(K_classes_cat, range_tuple=(1,3), max_features=30000, feature_type='KClassUniBiTri')

k_class_model_uni_bi_tri = MyMLP(input_features=num_features_k_class_uni_bi_tri, hidden_size=5000, num_classes=len(K_classes_cat)).to(device)

summary(model=k_class_model_uni_bi_tri,
        input_data=torch.randn(BATCH_SIZE, num_features_k_class_uni_bi_tri).to(device), # Example input data with the correct shape
        col_names = ["input_size", "output_size", "num_params", "trainable", "params_percent"],
        col_width=20,
        row_settings=["var_names"],
        depth = 1,
        device=device
        )

##Training the model for K Classes with unigram, bigram, and trigram features

k_class_criterion_uni_bi_tri = torch.nn.CrossEntropyLoss()
##changing learning rate to 0.001 for the unigram model
k_class_optimizer_uni_bi_tri = torch.optim.SGD(k_class_model_uni_bi_tri.parameters(), lr=0.001)

comparision_dict_k_class = epoch_wise_training(k_class_model_uni_bi_tri, train_k_class_loader_uni_bi_tri, val_k_class_loader_uni_bi_tri, k_class_criterion_uni_bi_tri, k_class_optimizer_uni_bi_tri, num_epochs=50, cat_type='KClassUniBiTri', comparision_dict=comparision_dict_k_class)


##Test Data Evaluation for K classes with unigram, bigram, and trigram features

test_data_evaluation(data_20newsgroups_k_class_vectorizer_uni_bi_tri, K_classes_cat, k_class_model_uni_bi_tri, k_class_criterion_uni_bi_tri)




# %%
compare_models(comparision_dict_k_class) 

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

def train_model_sklearn_2class(X_train, X_test, y_train, y_test, ngram_range=(1,1), feature="unigram"):
    vectorizer = CountVectorizer(ngram_range=ngram_range) # unigram features
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)


    for model in [LinearSVC(random_state=RANDOM_STATE), DecisionTreeClassifier(random_state=RANDOM_STATE), KNeighborsClassifier()]:

        results = results_initialize()

        acc, f1, precision, recall = 0, 0, 0, 0

        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()


        print(
            f"{model.__class__.__name__} {feature}:"
            f"Accuracy: {acc:.2f}, "
            f"F1 Score: {f1:.2f}, "
            f"Precision: {precision:.2f}, "
            f"Recall: {recall:.2f}, "
            f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}"
        )

        #Adding NA to simplify report comparision
        results["train_acc"].append(acc)
        results["train_f1"].append(f1)
        results["train_precision"].append(precision)
        results["train_recall"].append(recall)

        max_len = max(len(v) for v in results.values())
        for k in results:
            while len(results[k]) < max_len: # to make it of same length as the epoch-wise results of the neural network models for comparision
                results[k].append(float('nan'))
        


        comparision_dict_2class[model.__class__.__name__][feature] = copy.deepcopy(results)

# Load the dataset
data = fetch_20newsgroups(subset='all', categories=categories_list, shuffle=True, random_state=RANDOM_STATE)
corpus = data.data         # list of documents (raw text)
labels = data.target       # array of class labels

# Split into train and test (or train and validation)
X_train, X_test, y_train, y_test = train_test_split(
    corpus, labels, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)

train_model_sklearn_2class(X_train, X_test, y_train, y_test, ngram_range=(1,1), feature='2ClassUnigram') # unigram features
train_model_sklearn_2class(X_train, X_test, y_train, y_test, ngram_range=(1,2), feature='2ClassUniBigram') # unigram and bigram features
train_model_sklearn_2class(X_train, X_test, y_train, y_test, ngram_range=(1,3), feature="2ClassUniBiTrig") # unigram, bigram, and trigram features




# %%
compare_models(comparision_dict_2class)

# %% [markdown]
# 4. Classification techniques like SVM, decision tree, random forests, KNN for the multi-class classification problem and compute the train set, val set and test set accuracy, F1 score, Precision, recall and confusion matrix. Compare results with those obtained from feed forward network above.

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# K_classes_cat = ['sci.crypt', 'talk.politics.mideast', 'sci.space']

def train_model_sklearn_kclass(X_train, X_test, y_train, y_test, ngram_range=(1,1), feature=None):
    vectorizer = CountVectorizer(ngram_range=ngram_range) # unigram features
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)


    for model in [LinearSVC(random_state=RANDOM_STATE), DecisionTreeClassifier(random_state=RANDOM_STATE), KNeighborsClassifier(n_neighbors=2), RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)]:

        results = results_initialize()

        acc, f1, precision, recall = 0, 0, 0, 0

        epoch = 1

        
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')

        cm = confusion_matrix(y_test, y_pred)

        print(
            f"{model.__class__.__name__} {feature}:"
            f"Accuracy: {acc:.2f}, "
            f"F1 Score: {f1:.2f}, "
            f"Precision: {precision:.2f}, "
            f"Recall: {recall:.2f}, "
            f"Confusion Metrix:\n{cm}"
        )

        #Adding NA to simplify report comparision
        results["train_acc"].append(acc)
        results["train_f1"].append(f1)
        results["train_precision"].append(precision)
        results["train_recall"].append(recall)

        ## Adding NA to simplify report comparision
        max_len = max(len(v) for v in results.values())
        for k in results:
            while len(results[k]) < max_len: # to make it of same length as the epoch-wise results of the neural network models for comparision
                results[k].append(float('nan'))

        comparision_dict_k_class[model.__class__.__name__][feature] = copy.deepcopy(results)


# Load the dataset
data = fetch_20newsgroups(subset='all', categories=K_classes_cat, shuffle=True, random_state=RANDOM_STATE)
corpus = data.data         # list of documents (raw text)
labels = data.target       # array of class labels

# Split into train and test (or train and validation)
X_train, X_test, y_train, y_test = train_test_split(
    corpus, labels, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)

train_model_sklearn_kclass(X_train, X_test, y_train, y_test, ngram_range=(1,1), feature='KClassUnigram') # unigram features
train_model_sklearn_kclass(X_train, X_test, y_train, y_test, ngram_range=(1,2), feature="KClassUniBigram") # unigram and bigram features
train_model_sklearn_kclass(X_train, X_test, y_train, y_test, ngram_range=(1,3), feature="KClassUniBiTrig") # unigram, bigram, and trigram features



# %%
compare_models(comparision_dict_k_class)


