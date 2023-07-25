import os
from pathlib import Path

import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import gradio as gr
import time
from transformers import AutoTokenizer, GenerationConfig, TextStreamer, pipeline
questions_dir=Path("Microsoft_QA")
questions_dir.mkdir(exist_ok=True, parents=True)

def write_file(question, answer, file_path):
    text = f"""
    Q: {question}
    A: {answer}
    """.strip()
    with Path(questions_dir / file_path).open("w") as text_file:
      text_file.write(text)
write_file(
    question="What is Microsoft Q&A?",
    answer="""Microsoft Q&A is a Microsoft site where you can get fast access to
     questions about Microsoft technologies with Q&A, a global, community-driven
      platform for timely, high-quality technical answers.""".strip(),
    file_path="question_1.txt",
)
write_file(
    question="I saw you launched a new Q&A site on January 10, 2023. Why?",
    answer="""We know how important it's for you to have access to fast,
     accurate answers to questions about Microsoft technologies. The new site has improved
     workflows and the user interface is optimized for readability and efficiency. We also have a
    more integrated experience with the Microsoft Learn ecosystem. Check out all the changes in our release notes.""".strip(),
    file_path="question_2.txt",
)
write_file(
    question="What are the major benefits of Microsoft Q&A versus Stack Overflow?",
    answer="""We love Stack Overflow. We will continue supporting our customers who ask questions there. We also surface answers from Stack Overflow on Microsoft Q&A.

Stack Overflow has specific criteria about what questions are appropriate for the community whereas Microsoft Q&A has an open policy regarding this, where all questions about Microsoft technologies are welcomed. More importantly, via Microsoft Q&A we can create unique experiences that allow us to provide the highest level of support for our customers. It's hard to get a full picture of the customer who is asking a question on Stack Overflow. However, on Microsoft Q&A we can connect the asker to their actual product usage and support contract. This enables new opportunities to offer the highest quality support.""".strip(),
    file_path="question_3.txt",
)
write_file(
    question="What is the difference between Microsoft Q&A and Microsoft Tech Community?",
    answer="""Microsoft Q&A is first party experience for technical answers on Microsoft products. Developers and IT pros can solve problems with the help of community experts and Microsoft engineering. The experience includes a rich knowledge base which ingests content from MSDN Forums and Stack Overflow to help customers get the answers they need quickly.

Microsoft Tech Community provides a connection between customers, Microsoft and community experts to share best practice and ideas, receive supportive encouragement and learn through discussions and blogs across Azure and Microsoft 365 products and programs.""".strip(),
    file_path="question_4.txt",
)
write_file(
    question="I heard Microsoft Q&A only supports English. Is there any plan to support non-English?",
    answer="""Yes, one of the benefits of the new site we launched on January 10 is that we can start supporting other languages. Stay tuned for more news about that.""".strip(),
    file_path="question_5.txt",
)
write_file(
    question="How do I report a new feature or a bug for Microsoft Q&A?",
    answer="""Until we move to a unified product feedback, you can continue providing feedback about Q&A by using the In the meantime, you can provide us feedback by using the Microsoft Q&A tag.""".strip(),
    file_path="question_6.txt",
)
write_file(
    question="Where can I report offensive content?",
    answer="""Select the Report option next to the content you would like to report. From the drop-down box, select the reason for the offensive post. See more details in the report content article.

A moderator will review this report and act on it.""".strip(),
    file_path="question_7.txt",
)
write_file(
    question="Is Microsoft Q&A mobile-friendly?",
    answer="""The site is accessible and usable on mobile devices.""".strip(),
    file_path="question_8.txt",
)
write_file(
    question="How do I create an account?",
    answer="""If you have an account through Microsoft Learn, you can use the same account information here. If not, in the top right corner of the Q&A site you will see an option to sign in, which takes you to the process to create a new account. For more information, visit Managing your Microsoft Learn profile.""".strip(),
    file_path="question_9.txt",
)
write_file(
    question="How do I sign in?",
    answer="""In the Microsoft Q&A header, select Sign in. You can use the same user profile that you use on Microsoft Learn.""".strip(),
    file_path="question_10.txt",
)
write_file(
    question="Can other users see my email address?",
    answer="""No. Take a look at another user's profile and you'll notice that an email address is not listed. The same goes for users who look at your user profile.""".strip(),
    file_path="question_11.txt",
)
write_file(
    question="Why do I have to navigate to different pages to find the content that I am looking for instead of having infinite scrolling?",
    answer="""Infinite scrolling is not meant for our type of content. It has some issues such us: Difficulty re-finding content; Illusion of completeness; Inability to access the end of the page; Accessibility problems (users have to tab forever); Increased page load; Poor SEO performance.""".strip(),
    file_path="question_12.txt",
)
write_file(
    question="Why is there so much empty space on the right and left columns on the question threads? Can you expand the text to cover the full width of the page?",
    answer="""The design is to keep consistency across the Microsoft Learn site. It's design to improve readability to all users. Here are some resources that explains this in detail: [https://m.mediawiki.org/wiki/Reading/Web/Desktop_Improvements/Features/Limiting_content_width/]( Reading/Web/Desktop Improvements/Features/Limiting content width), [https://www.w3.org/WAI/WCAG21/Understanding/visual-presentation#dfn-blocks-of-text](Visual presentation of blocks of text), (https://baymard.com/blog/line-length-readability)[Line length readability].""".strip(),
    file_path="question_13.txt",
)
write_file(
    question="How do I ask a question?",
    answer="""In the Microsoft Q&A header, select the Ask a question button. As you type in your question, we will show you similar questions that have already been asked on Q&A and other sources. Make sure to check these because someone may have already answered your question.

You can find more details on best practices when asking questions in the How to write a quality question article.""".strip(),
    file_path="question_14.txt",
)

write_file(
    question="How do I know if a user is knowledgeable?",
    answer="""If a user participates in and contributes positively to the community frequently, they will win points. The more reputation points a user has, the more likely they're to have expertise on the topic they're posting.

We also show Microsoft affiliations in the user's cards: Microsoft Employee, Microsoft Agency Temporary, Microsoft Vendor, Microsoft Intern, and Microsoft Most Valuable Professionals (MVPs), so you know if the answer comes from a Microsoft-trusted source.""".strip(),
    file_path="question_16.txt",
)
write_file(
    question="How can I sort all the questions?",
    answer="""At the top right of the questions page, you can see a Sort by drop-down menu. Select the option you would like to sort the content by. You can also filter out the content based on different criteria. Check our filter and sort article for details.""".strip(),
    file_path="question_17.txt",
)

write_file(
    question="Who can answer a question?",
    answer="""Anyone signed in user can help others by answering a question on Microsoft Q&A. If you know the answer to a question, support the online community by answering it!""".strip(),
    file_path="question_19.txt",
)
write_file(
    question="Can I vote my own content?",
    answer="""You can't vote your own questions, comments, or answers.""".strip(),
    file_path="question_20.txt",
)
write_file(
    question="Can I vote on a question, answer, or comment more than once?",
    answer="""No, you can only vote once. You can change or cancel your vote if you change your mind.""".strip(),
    file_path="question_21.txt",
)
write_file(
    question="What are tags?",
    answer="""Tags are topics related to your thread that help group and organize all the content on Microsoft Q&A. You can add tags to any kind of post by searching from a wide range of topics in the question that you're creating.""".strip(),
    file_path="question_22.txt",
)
write_file(
    question="Do I have to add tags to a new question?",
    answer="""Yes, a minimum of 1 tag is required for each question. This helps the experts on that tag to monitor and answer your questions. The more tags you add the more information the community receives, making it easier to find similar questions and answer them.""".strip(),
    file_path="question_23.txt",
)
write_file(
    question="How do I follow a tag?",
    answer="""Following a tag allows you to get alerts on the tag related to a particular service. To follow a tag, select Tags on the header, and hover the tag you want to follow. Select the Follow button. You can also follow tags from other pages by hovering on the tag and selecting on the Follow button.

If you want to receive notifications when a question is posted on the tags you follow, select your Avatar, and then Settings. Make sure you have an email for notifications. Then, on Q&A preferences section, select Receive email when a question is posted on a tag you follow and Follow questions with tags you're following and select Save. You can find more details in the Q&A preferences article.""".strip(),
    file_path="question_24.txt",
)
write_file(
    question="Can I create a new tag?",
    answer="""Users can't create new tags at this moment.""".strip(),
    file_path="question_25.txt",
)
write_file(
    question="What are reputation points?",
    answer="""Reputation points are earned by participating in and contributing positively to the community. For more information on reputation points, see the reputation points help article.""".strip(),
    file_path="question_26.txt",
)
write_file(
    question="How can I manage my settings?",
    answer="""In the Microsoft Q&A header, select your avatar in the top right, then select Settings. This will take you to your profile page where you can provide a notification email address as well as your Microsoft Learn and Q&A settings. You can find more details on the Q&A preferences article.""".strip(),
    file_path="question_27.txt",
)
write_file(
    question="How do I update my profile?",
    answer="""In the Microsoft Q&A header, select your avatar in the top right and then select Profile. Here you will see options to update the different parts of your profile.""".strip(),
    file_path="question_28.txt",
)
write_file(
    question="How can I subscribe/unsubscribe to get email notifications on different kinds of threads?",
    answer="""In the Microsoft Q&A header, select your avatar on the top right, and then select Settings. Under Q&A preferences you can choose what kinds of threads on Microsoft Q&A you get notified about and how often. You can find more details on the Q&A preferences article.""".strip(),
    file_path="question_29.txt",
)
write_file(
    question="How can I see all the questions I save in a collection?",
    answer="""Select your avatar in the top right of the Microsoft Q&A header and then select Profile. Select Collections on the left section.""".strip(),
    file_path="question_30.txt",
)
write_file(
    question="How can I view the questions and answers I've interacted with?",
    answer="""Select your avatar in the top right of the Microsoft Q&A header and then select Profile. Under Activity you will see up to 30 interactions over the past 30 days..

""".strip(),
    file_path="question_31.txt",
)
write_file(
    question="What can moderators do?",
    answer="""Moderators are a part of the Microsoft Q&A community to maintain high quality content. Moderators can delete/undelete content, ban users, redirect threads, close threads, and edit content to keep threads relevant and appropriate.""".strip(),
    file_path="question_32.txt",
)
write_file(
    question="If a moderator's message gets flagged, can they cancel that report?",
    answer="""They can't, only another moderator will be able to.""".strip(),
    file_path="question_33.txt",
)
write_file(
    question="Can moderators mark answers as Accepted Answers?",
    answer="""No, only the question author have those permissions.""".strip(),
    file_path="question_34.txt",
)
write_file(
    question="What happens to the ongoing conversations in MSDN and TechNet forums that are closed to all new and existing threads?",
    answer="""You will still be able to view the forums, you just won't be able to ask new questions or create new responses. You can also see the questions surfaced in Microsoft Q&A whenever you ask a question or search on the site.""".strip(),
    file_path="question_35.txt",
)
write_file(
    question="Will the content from MSDN and TechNet forums be migrated into Microsoft Q&A?",
    answer="""No. When a user searches for something that doesn't appear when they're browsing on Microsoft Q&A, we use machine learning to display read-only questions and answers from MSDN and TechNet forums.""".strip(),
    file_path="question_36.txt",
)
write_file(
    question="Will I lose my MSDN and TechNet reputations?",
    answer="""Currently you can't carry over your MSDN and TechNet reputation. We are working to give you the opportunity to link Microsoft Q&A and MSDN and TechNet forums. When this is an option, your current badges and points from MSDN and TechNet forums will be displayed as part of your Microsoft Q&A profile.""".strip(),
    file_path="question_37.txt",
)
write_file(
    question="Can I keep the same user profile I have on MSDN and TechNet forums?",
    answer="""To use Microsoft Q&A you'll need to make a new profile (though you can keep your MSDN or TechNet identity). The new user profile you will create for Microsoft Q&A will be linked to Microsoft Learn. So if you already have an account for those sites, you can use the same one on Microsoft Q&A.""".strip(),
    file_path="question_38.txt",
)
write_file(
    question="I'm a moderator in MSDN and TechNet forums, will I be a moderator on Microsoft Q&A?",
    answer="""All moderators in MSDN and TechNet forums can continue being moderators. Follow the process described on the Microsoft Q&A moderators article.""".strip(),
    file_path="question_39.txt",
)
write_file(
    question="Where does Microsoft Q&A store customer data?",
    answer="""Microsoft Q&A doesn't move or store customer data out of the region it's deployed in.""".strip(),
    file_path="question_40.txt",
)
model_name = "TheBloke/Nous-Hermes-13B-GPTQ"
model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast= True)

model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    model_basename= model_basename,
    use_safetensors=True,
    Trust_remote_code=True,
)

generation_config = GenerationConfig.from_pretrained(model_name)
streamer = TextStreamer(
    tokenizer, skip_prompt = True, skip_special_tokens=True, use_multiprocessing = False
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer= tokenizer,
    max_length=2048,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    generation_config=generation_config,
    streamer=streamer,
    batch_size=1,

)
llm=HuggingFacePipeline(pipeline=pipe)
embeddings = HuggingFaceEmbeddings(
    model_name= 'embaas/sentence-transformers-multilingual-e5-base'

)
loader = DirectoryLoader("./Microsoft_QA/", glob="**/*txt")
documents = loader.load()
len(documents)
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
db = Chroma.from_documents(texts, embeddings)
template = """
### Instruction: You're a microsoft QA platform support agent who is talking to user giving them information about the platform. Use only the chat history and the following information
{context}
to answer in a helpful manner to the question. if you dont knoe the answer - say I am not able to solve this problem myself. I will connect you to one of our human chat operators who will surely be able to help you!.
Keep your replies short, compassionate and informative.
{chat_history}

### Input: {question}
### Responses:
""".strip()
prompt = PromptTemplate(input_variables=["context","question","chat_history"], template=template)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
     llm,
     db.as_retriever(),
     memory=memory,
     combine_docs_chain_kwargs= {"prompt": prompt},
     #return_source_documents= True,
     )


import gradio as gr
import time

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def user(user_message, history):
        return gr.update(value="", interactive=False), history + [[user_message, None]]

    def bot(history):
        response = qa(history[-1][0])
        response = response['answer']

        history[-1][1] = ""
        for character in response:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

    response = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
    demo.queue()
    demo.launch()