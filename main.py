from langchain.embeddings import VertexAIEmbeddings
from google.cloud import aiplatform
import os
import math
import numpy as np
from scipy.spatial.distance import cosine

# Get the current directory of your Python script
script_directory = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(script_directory, "tw-rd-de-bill.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Define your query texts and articles as string variables
query_texts = ["運動", "藝文", "財經", "政治"]
articles = {
    "Article A": "郭李兼任冬盟U23台灣培訓隊總教練，昨於洲際午場賽後趕回台北準備亞錦賽，他表示，礙於冬盟賽程與大專棒球聯賽重疊，培訓隊選派業餘球員參加冬盟，希望年輕選手把握與外隊交手的機會，明年陣容會再調整，除了選進大學球隊好手，也有機會徵召小聯盟旅外球員。",
    "Article B": "公鹿今天在主場迎戰拓荒者，也是里拉德（Damian Lillard）交易後首度遇上老東家，最終在他31分5籃板4助攻的帶領下贏球，不過比賽中出現了尷尬的一幕，就是里拉德在快攻無人防守的情況下，灌籃放槍。事情發生在第一節，當時里拉德成功抄截後獨自往前場跑，隊友也馬上把球甩給他，眼見沒人防守，里拉德決定來個單手灌籃，不過最終高度不夠沒灌進，還差點被籃框蓋火鍋。",
    "Article C": "陳雨菲昨收下今年第4冠，積分來到101646分，上升一位來到世界第2，正在養傷的山口茜下滑至第3，球后仍是南韓安洗瑩，台灣一姐戴資穎維持世界第4。另外目前也都因傷所擾的兩位前世界冠軍泰國依瑟儂、印度辛度，各下滑一位、分居第11、12位，在中國大師賽打進4強的南韓女將金佳恩上升2位來到生涯最佳的世界第13。",
    "Article D": "洋基隊休賽季有幾大補強目標，中外野手與先發輪值，資深記者海曼（Jon Heyman）在《紐約郵報》對條紋軍的現狀進行整理，透露洋基目前盤算。\n\n海曼指出，洋基喜歡前MVP重砲貝林傑（Cody Bellinger），但昨日同為《紐約郵報》的桑契斯（Mark W. Sanchez）點出，貝林傑在幾個數據上不利（擊球初速，預期打擊率）恐有一些疑慮，然而海曼認為，強擊球率更偏好易揮大棒的球員如蓋洛（Joey Gallo），但他認為貝林傑在2好球後會傾向將球打進場而非繼續強力揮擊，因此比較沒有這個疑慮。\n\n此外，洋基的外野選擇同時也持續鎖定教士強打者索托（Juan Soto），另外海曼也點名南韓24歲好手李政厚，不過他表示李政厚現在有大概20支球隊在關注他。\n\n而今年因為腦震盪導致狀態不佳的明星一壘手瑞佐（Anthony Rizzo），他的經紀人表示目前的狀態很棒。\n\n目前洋基另一個著眼點是先發輪值，他們傳出想要網羅日職最強投山本由伸，對舊將蒙哥馬利（Jordan Montgomery）也持續關注當中。",
    "Article E": "出生於莫斯科的藝術家安東．維克多將俄國宇宙論與日本神道及撿骨儀式串聯，探究死亡與生產性的關係﹔李亦凡藉由雕塑、繪畫與影像投影，重組奇特敘事﹔陳界仁以社會邊緣人為主體，開展面對新自由主義世界中凹陷出的異托邦現狀。白鉉真透過影像，以獨特語彙捕捉人生的生存狀態，同時展現創作者不斷摧毀又重建的創作歷程。展覽詳詢關美館官網。",
    "Article F": "雲門此次歐巡行程，接下來還有11月30至12月2日，於倫敦沙德勒之井劇院演出《毛月亮》；12月6、7日，在西班牙馬德里水道劇院演出《十三聲》；12月10日於法國坎城舞蹈節演出《十三聲》；12月15、16日，在德國法茲堡劇院演出《毛月亮》；12月20、21日，於西班牙特內里費音樂廳演出《十三聲》。詳詢雲門官網。",
    "Article G": "高雄兒童美術館「小阿法α大未來」新展登場，以海洋為核心，透過設計、插畫、攝影、雕刻、編織及手作等多元媒材，呈現不同海洋生物角色與主題，引領大小朋友穿越海洋的現在、瓶頸與未來，展開對海洋環境的豐富想像。策展團隊種籽設計的創作者們橫跨了平面、美術、插畫、動畫以及食藝設計等領域，他們以細膩的手法呈現出如同童話繪本奇幻風格的展覽空間，並特別邀請到台灣首位鯨豚攝影師金磊、台中老雕刻店陳彫刻處，以及藝術家王荷瑄的羊毛氈作品共同展出，希望以創意童趣的策展打開α世代對於海洋永續的關注。",
    "Article H": "南美館表示，TEFF歐洲影展（Taiwan European Film Festival）由歐洲經貿辦事處主辦，邀請19個歐洲國家各推選出1部電影參展，自 2005年開辦至今邁入第19年，此次南美館首次加入串聯播映據點之一，希望藉由帶領觀眾欣賞歐洲電影的過程，認識歐洲國家的文化、藝術和語言的多樣性。",
    "Article I": "預算中心引用審計部決算報告指出，依財政部賦稅署提供2019至2021年度綜所稅結算申報資料等，列報出租房屋或土地租賃收入者中，推估出租予家戶或個人介於32萬餘筆（2019及2020年）至33萬餘筆（2021年）間；惟主計總處2020年人口及住宅普查初步報告，家戶居住於租用住宅比率為10.94%，即應約有87.6萬戶，較32萬餘筆資料，存有明顯差距。",
    "Article J": "對於立委提案將房屋租金由「列舉扣除額」改為「特別扣除額」，財政部說明，房租屬納稅人滿足居住需要的重要支出，現行按實際支出列舉扣除，且與房貸利息擇一適用；但為落實居住正義，照顧租屋族，多位立委及黨團提案將房租支出改列特別扣除，政策立意良善，尊重委員提案。",
    "Article K": "法人認為，中信金10月單月稅後盈餘50.06億元，累計前10月稅後盈餘535.93億元、年增達59％，每股稅後盈餘（EPS）2.69元，是自2021年首度破500億元後，再創下歷史同期新高紀錄，且第3季單季稅後盈餘達198億元，更創下金控成立以來單季最高，金控全年獲利可望創歷史新高，獲得外資青睞。",
    "Article L": "日前根據國家規劃機構數據，由於出口和政府支出疲軟，泰國第3季經濟成長年增1.5%，低於預期，為今年最慢增速。賽塔過去是房地產大亨，他的目標是在未來4年，讓泰國年平均成長5%，過去十年平均成長率為1.9%，落後於東南亞地區其他國家。\n\n泰國央行行長蘇蒂瓦特納魯普（Sethaput Suthiwartnarueput）表示，泰國的財政和貨幣政策需要一定的空間，以確保經濟在成長過程中保持彈性。「彈性的要素是穩定、強勁的資產負債表、具有多種選擇的財政和貨幣政策。」",
    "Article M": "林聿禪出示LSE過去駁斥抹黑蔡總統學歷的聲明，以及其祝賀博士校友蔡英文連任台灣總統的公告指出，無論是蔡總統的博士學位，或2011年6月拜會倫敦政經學院的行程，凡有基本數位能力的現代人，都可以經由上網輕鬆查到正確訊息，馬前總統以一位前任總統的地位，若因助選選擇造謠，將令人感到相當的遺憾。",
    "Article N": "除了趙少康提出棄柯保侯，侯友宜昨天上午輔選北市立委參選人游淑慧時致詞表示，游願意到士林大同艱困選區參選，「很不簡單，人家說她選不上，但我保證她一定上」，「勇敢的人、不怕死的人，遇到不要臉的人，不要臉的一定死，不怕死才會勝利」。",
    "Article O": "不過趙少康身為中廣董事長，本身還有在主持政論性談話節目，外界質疑是否有違反「廣播電視法」規定，以及相關媒體附款規定。國家通訊傳播委員會（NCC）副主任委員翁柏宗表示，趙少康受政黨推薦登記為正副總統候選人，是否違反黨政軍條款，NCC委員會請趙少康、中廣、TVBS提出說明，後續依程序去認定。\n\n翁柏宗表示，趙少康受政黨推薦登記為正副總統候選人，是否有違反廣播電視法第5條之1黨政軍條款規定，「政府、政黨、政黨黨務工作人員，及選任公職人員不得投資」，及「不得擔任董監事等規定」，這部分尚待NCC委員會認定是否有違反情況，後續也會請趙少康陳述意見。另外，2007年中廣股權交易案時，有向NCC提出「經營不會受黨政軍影響」的承諾，以及今年董監事變更案時所列附款，後續NCC均會檢視，並請中廣說明趙少康主持節目的情況。",
    "Article P": "時代力量新竹縣第2選區立委王婉諭今天以「有婉諭，健康有保障」為題，邀請伯拉罕共生照顧勞動合作社理事主席、時力不分區立委候選人林依瑩共同宣布她的「水污染防治、醫療、長照」面向政見。林依瑩允諾只要王婉諭當選，她會以過去推動「不老」系列的銀髮族創意樂活活動，與台中市和平區達觀部落透過柏拉罕共生長照成功解決當地老年與重症照顧需求，並促成部落青年返鄉就業的經驗，協助新竹原鄉複製、發展嘉惠鄉親。",
}


# Create an embeddings model
embeddings_model = VertexAIEmbeddings(model_name="textembedding-gecko-multilingual")

# Create an empty dictionary to store cosine similarity scores
similarity_scores = {}

# Embed query texts
embedded_queries = [
    embeddings_model.embed_query(query_text) for query_text in query_texts
]

# Calculate cosine similarities for each article and each category
for article_name, article_text in articles.items():
    embedded_article = embeddings_model.embed_query(article_text)
    scores_for_article = []

    for query_embed in embedded_queries:
        similarity_score = 1 - cosine(embedded_article, query_embed)
        scores_for_article.append(similarity_score)

    similarity_scores[article_name] = scores_for_article

# Print details
for article_name, scores in similarity_scores.items():
    print(f"Article: {article_name}")
    for i, query_text in enumerate(query_texts):
        print(f"Similarity to '{query_text}': {scores[i]:.4f}")

# Resulting dictionary with similarity scores
print("\nSimilarity Scores:")
print(similarity_scores)


# def modified_sigmoid(x, average):
#     k = 20
#     return 1 / (1 + math.exp(-k * (x - average)))


# # Assuming similarity_scores is your dictionary with the scores
# # Calculate the average for each article
# averages = {
#     article: sum(scores) / len(scores) for article, scores in similarity_scores.items()
# }

# # Apply the modified sigmoid function to each score
# sigmoid_scores = {}
# for article, scores in similarity_scores.items():
#     average = averages[article]
#     sigmoid_scores[article] = [modified_sigmoid(score, average) for score in scores]

# # Print the modified sigmoid scores for each article
# for article, scores in sigmoid_scores.items():
#     print(f"Modified Sigmoid Scores for {article}: {scores}")

# Define user-article interactions (fake data)
user_interactions = {
    "Alice": {
        "Article A": 8,
        "Article B": 1,
        "Article C": 3,
        "Article D": 2,
        "Article E": 5,
        "Article F": 7,
        "Article G": 2,
        "Article H": 4,
        "Article I": 6,
        "Article J": 1,
        "Article K": 3,
        "Article L": 9,
        "Article M": 8,
        "Article N": 2,
        "Article O": 4,
        "Article P": 6,
    },
    "Bob": {
        "Article A": 1,
        "Article B": 7,
        "Article C": 1,
        "Article D": 3,
        "Article E": 2,
        "Article F": 8,
        "Article G": 6,
        "Article H": 5,
        "Article I": 3,
        "Article J": 7,
        "Article K": 2,
        "Article L": 1,
        "Article M": 4,
        "Article N": 5,
        "Article O": 9,
        "Article P": 3,
    },
    "Ray": {
        "Article A": 3,
        "Article B": 2,
        "Article C": 5,
        "Article D": 1,
        "Article E": 8,
        "Article F": 1,
        "Article G": 9,
        "Article H": 7,
        "Article I": 2,
        "Article J": 4,
        "Article K": 6,
        "Article L": 3,
        "Article M": 5,
        "Article N": 7,
        "Article O": 8,
        "Article P": 2,
    },
    "Eva": {
        "Article A": 6,
        "Article B": 3,
        "Article C": 4,
        "Article D": 7,
        "Article E": 1,
        "Article F": 5,
        "Article G": 3,
        "Article H": 2,
        "Article I": 7,
        "Article J": 5,
        "Article K": 8,
        "Article L": 4,
        "Article M": 6,
        "Article N": 1,
        "Article O": 3,
        "Article P": 9,
    },
    "Max": {
        "Article A": 2,
        "Article B": 4,
        "Article C": 7,
        "Article D": 5,
        "Article E": 6,
        "Article F": 2,
        "Article G": 8,
        "Article H": 1,
        "Article I": 4,
        "Article J": 6,
        "Article K": 9,
        "Article L": 5,
        "Article M": 3,
        "Article N": 8,
        "Article O": 7,
        "Article P": 1,
    },
    "Luna": {
        "Article A": 9,
        "Article B": 6,
        "Article C": 2,
        "Article D": 8,
        "Article E": 4,
        "Article F": 9,
        "Article G": 1,
        "Article H": 8,
        "Article I": 5,
        "Article J": 3,
        "Article K": 7,
        "Article L": 6,
        "Article M": 2,
        "Article N": 9,
        "Article O": 5,
        "Article P": 4,
    },
}


# Calculate user scores based on interactions
user_scores = {}

for user, interactions in user_interactions.items():
    user_scores[user] = [0, 0, 0, 0]  # Initialize user scores for each category
    for article, count in interactions.items():
        article_scores = similarity_scores[article]
        user_scores[user] = [
            s + (count * article_scores[i]) for i, s in enumerate(user_scores[user])
        ]

# Apply softmax normalization to user scores
softmax_normalized_scores = {}

for user, scores in user_scores.items():
    total_score = sum(scores)
    softmax_scores = [s / total_score for s in scores]
    softmax_normalized_scores[user] = softmax_scores

# Print user scores and softmax normalized scores for debugging
print("User Scores:")
for user, scores in user_scores.items():
    print(f"{user}: {scores}")

print("\nSoftmax Normalized Scores:")
for user, scores in softmax_normalized_scores.items():
    print(f"{user}: {scores}")

categories = ["運動", "美術", "財經", "政治"]

# Find the person with the maximum value for each category
max_person_for_category = {}

for i, category in enumerate(categories):
    max_person = max(
        softmax_normalized_scores,
        key=lambda person: softmax_normalized_scores[person][i],
    )
    max_score = softmax_normalized_scores[max_person][i]
    max_person_for_category[category] = (max_person, max_score)

# Print the max person and their score for each category
for category, (person, score) in max_person_for_category.items():
    print(f"Max value for '{category}': {person} with score {score:.4f}")
