INSTRUCTION = '''First, generate your thought process aloud in first-person in between the <think> tags. Then, outside of the <think> tag, generate your final answer. You must stop thinking outside of the <think> tag, and commit to providing an answer in the format of **Answer**: $ANSWER. Here are some examples.

{example1}

{example2}

{example3}

This is the end of the examples. The following is the question you need to answer following the style of the examples.

'''

EXAMPLE1 = '''
[Example 1]
Question: 
Who was the next British Prime Minister after Arthur Balfour?
Output:
<think>\nOkay, so I need to figure out who was the next British Prime Minister after Arthur Balfour. I'm not too familiar with British Prime Ministers, especially from that time period, so I'll have to think this through step by step.\n\nFirst, I know that Arthur Balfour was a Prime Minister, but I'm not exactly sure when he served. I think it was in the early 20th century. Maybe around World War I? I remember that David Lloyd George was a Prime Minister during that time, but I'm not sure if he came after Balfour or before.\n\nWait, I think Balfour was Prime Minister before the war. Let me try to recall the order. There was Herbert Asquith, who was Prime Minister before the war. Then, during the war, Lloyd George took over. So, if Balfour was before Lloyd George, then he must have been Prime Minister before Asquith? Or was it the other way around?\n\nI'm getting a bit confused. Let me try to list the Prime Ministers around that time. I know that before Balfour, there was William Ewart Gladstone and then Robert Cecil, Lord Salisbury. After Salisbury, there was Arthur Balfour. So, Balfour was Prime Minister from 1902 to 1905, I think. Then, after him, Herbert Asquith took over in 1908, right? Wait, no, that doesn't make sense because there was a gap between 1905 and 1908. Maybe Henry Campbell-Bannerman was Prime Minister in between.\n\nYes, I think Henry Campbell-Bannerman was Prime Minister from 1905 to 1908. So, after Balfour, it was Campbell-Bannerman, and then Asquith. So, the next Prime Minister after Balfour would be Campbell-Bannerman.\n\nWait, but I'm not entirely sure. Let me think again. Balfour resigned in 1905, and then Campbell-Bannerman formed a government. So, yes, he was the next one. Then, when Campbell-Bannerman resigned in 1908, Asquith took over. So, the sequence is Balfour, then Campbell-Bannerman, then Asquith, and then Lloyd George.\n\nI think that's correct. So, the next Prime Minister after Arthur Balfour was Henry Campbell-Bannerman.\n</think>\n\n**Answer**: Henry Campbell-Bannerman
[Example 1 End]
'''


EXAMPLE2 = '''
[Example 2]
Question: 
Men Against the Sea and Pitcairn's Island were two sequels to what famous novel?
Output:
<think>\nOkay, so I need to figure out which famous novel \"Men Against the Sea\" and \"Pitcairn's Island\" are sequels to. Hmm, I'm not entirely sure, but I'll try to break it down step by step.\n\nFirst, I remember that \"Men Against the Sea\" sounds familiar. I think it's a book, maybe a classic. The title makes me think of adventure or survival themes. \"Pitcairn's Island\" also rings a bell; I think it's a real place, a remote island in the Pacific. Maybe the novel is about people stranded there?\n\nWait, I think there's a famous novel about a group of boys who are stranded on an island. Oh, right! \"Lord of the Flies\" by William Golding. But wait, is that the one? Or is it another book? I'm a bit confused because I know \"Lord of the Flies\" is about boys surviving on an island, but I'm not sure if it has sequels.\n\nAlternatively, I recall that there's a series called \"The Swiss Family Robinson.\" That's about a family surviving on a deserted island. But I don't think \"Men Against the Sea\" is part of that series. Maybe it's a different book.\n\nWait, another thought: there's a book called \"The Coral Island\" by R.M. Ballantyne. It's about three boys stranded on a coral island. But again, I'm not sure if \"Men Against the Sea\" is a sequel to that.\n\nLet me think about the titles again. \"Men Against the Sea\" sounds like it could be a sequel to a story where people are dealing with the sea, maybe after a shipwreck. \"Pitcairn's Island\" is a real place, known for the mutiny on the Bounty. Oh! The mutiny on the Bounty! There's a famous novel about that, \"Mutiny on the Bounty\" by James Norman Hall and Charles Nordhoff. But wait, is that the original novel, and are \"Men Against the Sea\" and \"Pitcairn's Island\" sequels?\n\nI think \"Mutiny on the Bounty\" is the first book, and then there are sequels. Let me check: the first book is about the mutiny, then the second is about the mutineers settling on Pitcairn's Island, and the third might be about their descendants. So, \"Men Against the Sea\" could be the second book, and \"Pitcairn's Island\" the third. Or maybe the other way around.\n\nWait, I'm getting a bit mixed up. Let me try to recall the order. The first book is \"Mutiny on the Bounty,\" then the second is \"Men Against the Sea,\" which follows the mutineers after the mutiny, and the third is \"Pitcairn's Island,\" which deals with the descendants of the mutineers. So, yes, both \"Men Against the Sea\" and \"Pitcairn's Island\" are sequels to \"Mutiny on the Bounty.\"\n\nI think that's it. So the famous novel they're sequels to is \"Mutiny on the Bounty.\" I'm pretty confident now because I remember the series being about the mutiny and its aftermath, including the settlement on Pitcairn's Island.\n</think>\n\n**Answer**: Mutiny on the Bounty.
[Example 2 End]
'''

EXAMPLE3 = '''
[Example 3]
Question: 
River Phoenix died during the making of which movie?
Output:
<think>\nOkay, so I need to figure out what the Paramount Film Company was originally called. I'm not too familiar with the history of movie studios, but I'll try to think it through step by step.\n\nFirst, I know that Paramount is one of the major Hollywood studios. I've heard the term \"Paramount\" before, but I'm not sure about its origins. Maybe it started with a different name and then changed to Paramount later.\n\nI remember that many old movie studios were founded by different people or companies. For example, Warner Bros. was started by the Warner brothers, and MGM stands for Metro-Goldwyn-Mayer, which was a merger of several companies. So perhaps Paramount also came from a merger or a rebranding.\n\nI think the name \"Paramount\" suggests something important or superior, which makes sense for a film studio aiming to be top-tier. But I'm not sure when they started using that name. Maybe it was in the early 20th century when the film industry was booming.\n\nI recall that in the early days of Hollywood, there were several independent film companies. One of them was called the \"Hartford\" something. Wait, maybe it was the \"Hartford Film Company\"? Or perhaps it was \"Hartford\" as part of a longer name. I'm not entirely sure, but I think there's a connection there.\n\nAnother thought: I've heard the name Adolph Zukor in relation to Paramount. He was a key figure in the studio's history. Maybe he was involved in renaming the company. So, if Adolph Zukor was involved, perhaps the original name was something else before he took over or rebranded it.\n\nPutting it together, I think the original name might have been the \"Hartford Film Company,\" and then it was renamed to Paramount. But I'm not 100% certain. I should probably check some facts to confirm this.\n\nWait, I think the original name was actually the \"Hartford\" something else. Maybe it was the \"Hartford\" as part of a longer name, like \"Hartford-Famous Players\" or something similar. I'm a bit fuzzy on the exact details, but I believe it was a merger or a rebranding that led to the name change to Paramount.\n\nSo, to sum up my thoughts: Paramount was originally called the \"Hartford Film Company,\" but I'm not entirely sure. It might have been a different name, perhaps involving \"Hartford\" and another term, and Adolph Zukor played a role in the rebranding. I think the key point is that it was renamed to Paramount, likely in the early 1900s, to signify its importance in the film industry.\n\n</think>\n\n**Answer**: Hartford Film Company.
[Example 3 End]
'''


PROMPT = INSTRUCTION.format(example1=EXAMPLE1, example2=EXAMPLE2, example3=EXAMPLE3)
