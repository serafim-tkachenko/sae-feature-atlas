# Automated inspection report: `gemma31bpt_l13_res16k_pile10k_top32`

This is a heuristic triage report. Labels are not final semantic interpretations.

## What this report tries to catch

- formatting/tokenization artifacts
- quote/punctuation-heavy features
- boundary-position-heavy features
- token-concentrated features
- suspicious coactivation cliques
- candidate features for manual semantic inspection

Note: `source_concentrated` is only used when the run contains multiple source categories.

## Top likely artifact features

### Feature 16372
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.520`
- Semantic score: `0.203`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.13`

Top tokens:
```text
{
  "\n": 91,
  "\n\n": 2,
  "    ": 1
}
```
Example:
```text
activation=209.36, source=pile-10k, text_id=121, token_pos=247
graphics, cx, cy, spikes, outerRadius, innerRadius, color, lineColor)
{[
]    var rot = Math.PI /
```

### Feature 4044
- Labels: `quote_like, punctuation_like, likely_artifact`
- Artifact score: `0.520`
- Semantic score: `0.395`
- Quote / punctuation / boundary share: `0.97` / `1.00` / `0.06`

Top tokens:
```text
{
  " \"": 37,
  " “": 33,
  " '": 7,
  "“": 5,
  "«": 2,
  " `": 1,
  " ‘": 1,
  "‘": 1,
  "—": 1
}
```
Example:
```text
activation=227.89, source=pile-10k, text_id=60, token_pos=247
 lawmaker for his conspicuous wealth.

“What a wonderful country we have,” said the former mayor.[ “]The best-known socialist in the country
```

### Feature 13402
- Labels: `punctuation_like, space_like, token_concentrated, position_concentrated, likely_artifact`
- Artifact score: `0.513`
- Semantic score: `0.209`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.04`

Top tokens:
```text
{
  "\n\n": 71,
  "\n": 3
}
```
Example:
```text
activation=223.71, source=pile-10k, text_id=80, token_pos=121
 for this conference are Mark Huckvale and Jeff M. Smith. This marks…
View this post[

]From the archives of the late, great Recording Engineer/Producer (RE/P) magazine, enjoy
```

### Feature 1480
- Labels: `quote_like, punctuation_like, token_concentrated, likely_artifact`
- Artifact score: `0.513`
- Semantic score: `0.352`
- Quote / punctuation / boundary share: `0.91` / `0.93` / `0.04`

Top tokens:
```text
{
  "'": 42,
  "’": 23,
  "´": 2,
  "s": 2,
  " '": 2,
  "0": 2,
  "2": 1,
  "‘": 1,
  "''": 1
}
```
Example:
```text
activation=315.93, source=pile-10k, text_id=75, token_pos=36
've got a Doctrine2 Entity called "Order", which has several status properties. The allowed status['] are stored in a different Entity, so there is a ManyToOne relation defined for those entities.

```

### Feature 2705
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.512`
- Semantic score: `0.238`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.17`

Top tokens:
```text
{
  "\n": 93,
  "\r": 6,
  "\n\n": 6,
  "\n\n\n": 1
}
```
Example:
```text
activation=188.73, source=pile-10k, text_id=273, token_pos=240
id="selCampID" class="form-control" ng-model="campInput" >[
]            <option ng-repeat="camp in campaigns" value="{{camp.
```

### Feature 1657
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.510`
- Semantic score: `0.196`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.05`

Top tokens:
```text
{
  "\n": 647
}
```
Example:
```text
activation=445.00, source=pile-10k, text_id=118, token_pos=92
 distortion: -

See also this article, Crossover Distortion in Amplifiers, for more information.[
]Rv modifies the volt drop across the two series diodes. Remember that diodes are not just fixed 0
```

### Feature 2636
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.508`
- Semantic score: `0.205`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.07`

Top tokens:
```text
{
  " ": 79,
  " (": 1,
  "-": 1
}
```
Example:
```text
activation=218.46, source=pile-10k, text_id=44, token_pos=47
an

JACKSON HEIGHTS — Indian grocery store Patel Brothers is planning a renovation and expansion at its[ ]74th Street location that will feature a larger produce section and wider aisles, according to its manager
```

### Feature 229
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.506`
- Semantic score: `0.199`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.04`

Top tokens:
```text
{
  " ": 465,
  "-": 2
}
```
Example:
```text
activation=1022.74, source=pile-10k, text_id=92, token_pos=91
 to take a look at my code. But mostly experienced programmers don't have time for this.[ ]
So can I ask such questions on Stack Overflow?

A:

So can I ask such
```

### Feature 1010
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.506`
- Semantic score: `0.198`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.03`

Top tokens:
```text
{
  " ": 226
}
```
Example:
```text
activation=273.24, source=pile-10k, text_id=207, token_pos=80


Kering, which owns the luxury fashion brands Gucci and Saint Laurent, launched the legal action in[ ]2015 accusing Alibaba of being complicit in the sale of counterfeit goods on its websites.
```

### Feature 3419
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.504`
- Semantic score: `0.221`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.07`

Top tokens:
```text
{
  "\n": 89,
  "\n\n": 3,
  "\t": 3
}
```
Example:
```text
activation=342.42, source=pile-10k, text_id=140, token_pos=79
1</a></li>
       <li title="cust 2"></li>
       </ul>
     </td>[
]   </tr>
 </tbody>

 
when i do on below on IE 7, DOM element
```

### Feature 1328
- Labels: `quote_like, punctuation_like, likely_artifact`
- Artifact score: `0.502`
- Semantic score: `0.376`
- Quote / punctuation / boundary share: `0.84` / `1.00` / `0.05`

Top tokens:
```text
{
  "\"": 72,
  "'": 31,
  " \"": 8,
  "”": 5,
  " \"\"": 5,
  ".'": 3,
  "’": 2,
  "`": 2,
  ";\";": 2,
  "''": 2
}
```
Example:
```text
activation=466.21, source=pile-10k, text_id=173, token_pos=249
;";
        //string connectionstring = String.Format("Server={0};Port={1};["] +
        //       "
```

### Feature 6615
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.502`
- Semantic score: `0.221`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.06`

Top tokens:
```text
{
  "\n": 59,
  "\n\n": 4
}
```
Example:
```text
activation=278.78, source=pile-10k, text_id=203, token_pos=101
 may have its genesis in the
aerospace industry, but it has launched breakthrough designs in other industries[
]as well. Case in point: the medical industry, where device manufacturers have
used CAD and FE
```

### Feature 15680
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.502`
- Semantic score: `0.210`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.03`

Top tokens:
```text
{
  "\n\n": 65,
  "\n": 2
}
```
Example:
```text
activation=177.29, source=pile-10k, text_id=39, token_pos=74
 the map)



Thanks for stopping by, if you enjoyed this submission please consider leaving a diamond![

]The sweeping sunset of the Roughhew Rocks is deceiving. The mountain becomes beautiful, the forests become picturesque
```

### Feature 11323
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.502`
- Semantic score: `0.199`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.00`

Top tokens:
```text
{
  "\n": 52
}
```
Example:
```text
activation=268.93, source=pile-10k, text_id=138, token_pos=73
 is a strong relation between this problem and the same problem for the spaces $X_i$.'[
]address: 'IMAS-CONICET'
author:
- Jorge Tomás Rodríguez
title
```

### Feature 10759
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.501`
- Semantic score: `0.211`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.03`

Top tokens:
```text
{
  "\n\n": 87,
  "\n": 3
}
```
Example:
```text
activation=219.36, source=pile-10k, text_id=248, token_pos=89
 distinguish himself by wearing headbands. Tyrell is bulkier and less flashy with his wardrobe choices.[

]And when the dual-sport athletes take the football field, their differences continue to stack up.
```

### Feature 125
- Labels: `quote_like, punctuation_like, likely_artifact`
- Artifact score: `0.501`
- Semantic score: `0.400`
- Quote / punctuation / boundary share: `0.77` / `0.94` / `0.20`

Top tokens:
```text
{
  "'": 30,
  "’": 24,
  " ": 5,
  ".": 5,
  "/": 2,
  "1": 2,
  "0": 2
}
```
Example:
```text
activation=1832.55, source=pile-10k, text_id=57, token_pos=251
 a better idea. I was clearly a child genius.

I hated the entire night. I hadn[’]t been around for
```

### Feature 3105
- Labels: `punctuation_like, space_like, token_concentrated, position_concentrated, likely_artifact`
- Artifact score: `0.496`
- Semantic score: `0.232`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.02`

Top tokens:
```text
{
  "\n\n": 95,
  "\n": 4,
  "\r": 3,
  ":": 1,
  "\n\n\n\n": 1
}
```
Example:
```text
activation=164.18, source=pile-10k, text_id=162, token_pos=106
 following statements is CORRECT?

a. The company’s current stock price is $20.[

]b. The company’s dividend yield 5 years from now is expected to be 10
```

### Feature 10059
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.495`
- Semantic score: `0.216`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.00`

Top tokens:
```text
{
  "\n\n": 49,
  "\n\n\n\n": 1,
  "\n": 1
}
```
Example:
```text
activation=349.66, source=pile-10k, text_id=36, token_pos=25
NewYork) — A fight between two roommates inside a homeless shelter in Harlem turned deadly Friday morning.[

]It happened inside a shelter for men with alcohol, drug and mental health problems at 149
```

### Feature 9961
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.494`
- Semantic score: `0.234`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.05`

Top tokens:
```text
{
  "\n": 130,
  "\n\n": 13
}
```
Example:
```text
activation=267.10, source=pile-10k, text_id=93, token_pos=88
cormorantsc",
      using:      :svn,
      trust_cert: true[
]  name "Cormorant SC"
  homepage "https://fonts.google.com/
```

### Feature 2980
- Labels: `punctuation_like, space_like, token_concentrated, likely_artifact`
- Artifact score: `0.494`
- Semantic score: `0.246`
- Quote / punctuation / boundary share: `0.00` / `1.00` / `0.08`

Top tokens:
```text
{
  "\n\n": 105,
  "\n": 10,
  "\n\n\n\n": 2,
  "  ": 1,
  "\n\n\n\n\n\n": 1,
  "«": 1
}
```
Example:
```text
activation=307.28, source=pile-10k, text_id=34, token_pos=79
ke Eikeri and Tereza Mrdeža retired in the final at 1–0.[

]Seeds

Draw

References
Main Draw

XIXO Ladies Open Hódmezővás
```

## Top inspection candidates

### Feature 6887
- Semantic score: `0.992`
- Artifact score: `0.005`
- Labels: `inspection_candidate`

Example:
```text
activation=226.93, source=pile-10k, text_id=177, token_pos=180
 the infamous cable car going down Powell Street? MobileMuni is a complete guide to getting around San[ Francis]cos transit service that lets you know when busses or street cars will be arriving as well as assisting you
```

### Feature 178
- Semantic score: `0.991`
- Artifact score: `0.012`
- Labels: `inspection_candidate`

Example:
```text
activation=1170.76, source=pile-10k, text_id=38, token_pos=39
 gifts and charms, but H & M do resemble me in various ways :)

I usually like to[ write] stories from a single point of view. It’s obviously a limited perspective, but I enjoy the
```

### Feature 479
- Semantic score: `0.991`
- Artifact score: `0.012`
- Labels: `inspection_candidate`

Example:
```text
activation=955.50, source=pile-10k, text_id=116, token_pos=54
 Monday, Iowans will gather to launch the 2016 presidential election with an arcane[ ritual] — the caucus.

In living rooms and meeting halls throughout the state, caucus-goers will group
```

### Feature 234
- Semantic score: `0.990`
- Artifact score: `0.012`
- Labels: `inspection_candidate`

Example:
```text
activation=1096.40, source=pile-10k, text_id=6, token_pos=85
 best achieved by treating them the same as men or by making provisions that recognise their differences in terms of[ physiological] constitution and biological functions?

If the UK introduces such an initiative, it would not be the first
```

### Feature 241
- Semantic score: `0.990`
- Artifact score: `0.010`
- Labels: `inspection_candidate`

Example:
```text
activation=895.79, source=pile-10k, text_id=217, token_pos=28
 first post on the board, and I'm glad to see there is a section specifically on Spanish[ wines] as they've always been a favorite of mine!
I recently drank the wine mentioned in the
```

### Feature 233
- Semantic score: `0.989`
- Artifact score: `0.011`
- Labels: `inspection_candidate`

Example:
```text
activation=664.56, source=pile-10k, text_id=116, token_pos=62
 launch the 2016 presidential election with an arcane ritual — the caucus.

In living[ rooms] and meeting halls throughout the state, caucus-goers will group themselves into clusters according to which presidential candidate
```

### Feature 35
- Semantic score: `0.989`
- Artifact score: `0.011`
- Labels: `inspection_candidate`

Example:
```text
activation=550.15, source=pile-10k, text_id=27, token_pos=154
 coming one way or another!

Two weeks before my due date, I noticed some blood. My[ water] didn’t break and I saw no mucous plug, but it seemed that something was happening earlier than
```

### Feature 517
- Semantic score: `0.989`
- Artifact score: `0.013`
- Labels: `inspection_candidate`

Example:
```text
activation=729.24, source=pile-10k, text_id=117, token_pos=86
 fully electric crossover SUV that will wear the Mustang's chrome pony.

Ford has apparently learned from[ brands] like Porsche, Lamborghini and Jeep that even ardent fans will forgive the use of a brand name on a
```

### Feature 1407
- Semantic score: `0.989`
- Artifact score: `0.013`
- Labels: `inspection_candidate`

Example:
```text
activation=411.03, source=pile-10k, text_id=36, token_pos=103
 kitchen knives, CBS2’s Brian Conybeare reported.

Police say the two roommates[ pulled] out kitchen knives during a dipute. A 44-year-old was killed, and
```

### Feature 1720
- Semantic score: `0.989`
- Artifact score: `0.014`
- Labels: `inspection_candidate`

Example:
```text
activation=309.53, source=pile-10k, text_id=79, token_pos=60
 compile and debug C,C++ programs in Notepad++.
System Details: (a) Turbo C[ directory] C:\TC (b) OS Windows 7
Please provide complete details on how to set Environment
```

### Feature 268
- Semantic score: `0.989`
- Artifact score: `0.014`
- Labels: `inspection_candidate`

Example:
```text
activation=992.69, source=pile-10k, text_id=253, token_pos=73
 Key challenges in this area include developing devices in which the molecular integrity is preserved, developing in situ character[ization] techniques to probe the molecules within the completed devices, and determining the physical processes that influence carrier transport.
```

### Feature 1671
- Semantic score: `0.989`
- Artifact score: `0.011`
- Labels: `inspection_candidate`

Example:
```text
activation=208.69, source=pile-10k, text_id=112, token_pos=181
5pm

Personal Tax Return for each partner (includes partnership income and bank interest received)

Limited[ Co].from £65pm

Year End Accounts

Accounts Filed at Companies House

Company Tax Return
```

### Feature 2605
- Semantic score: `0.988`
- Artifact score: `0.012`
- Labels: `inspection_candidate`

Example:
```text
activation=173.17, source=pile-10k, text_id=41, token_pos=192
 a North American and is a genuine made-in-China co-venture involving China Film Group,[ Wanda] Media, Village Roadshow Pictures Asia, and Universal Pictures.

The contemporary action-drama was shot
```

### Feature 1036
- Semantic score: `0.988`
- Artifact score: `0.014`
- Labels: `inspection_candidate`

Example:
```text
activation=208.69, source=pile-10k, text_id=18, token_pos=90
 Instead, jOOQ uses a reverse engineering paradigm (as in mapping relational entities to objects > "[ROM]").

Re: jOOQ on The ORM Foundation?

Object Role Modeling (the original
```

### Feature 6245
- Semantic score: `0.988`
- Artifact score: `0.009`
- Labels: `inspection_candidate`

Example:
```text
activation=243.10, source=pile-10k, text_id=135, token_pos=206
 of most cell phones, the core functionality has not seen a similar expansion. The reasons for the development[ discrepancy] likely have to do with the fact that the core functionality is sufficient for most users and that there are
```

### Feature 427
- Semantic score: `0.988`
- Artifact score: `0.015`
- Labels: `inspection_candidate`

Example:
```text
activation=619.66, source=pile-10k, text_id=251, token_pos=167
 phones.

Partygoers are being urged to be careful of their belongings this weekend, as people finish[ work] for the festive break. Today is widely known as Mad Friday — traditionally the last working Friday before Christmas
```

### Feature 2379
- Semantic score: `0.988`
- Artifact score: `0.011`
- Labels: `inspection_candidate`

Example:
```text
activation=152.04, source=pile-10k, text_id=115, token_pos=249
In deciding a motion for reconsideration, we examine whether the motion is
based upon newly discovered evidence,[ mistakes] in our findings of fact,
```

### Feature 5
- Semantic score: `0.988`
- Artifact score: `0.016`
- Labels: `inspection_candidate`

Example:
```text
activation=816.17, source=pile-10k, text_id=235, token_pos=18
<bos>Mind The Gap

America’s British population has taken to the web to voice its[ displeasure] at news that U.S. candy giant Hershey has successfully blocked our much loved U.K.-
```

### Feature 632
- Semantic score: `0.988`
- Artifact score: `0.014`
- Labels: `inspection_candidate`

Example:
```text
activation=608.82, source=pile-10k, text_id=169, token_pos=99
 contact us for more information about our products and services.
Read more

... house.
The[ value] of a property increases with the addition of a hydraulic elevator, electric home elevator, vacuum elevator or high
```

### Feature 232
- Semantic score: `0.988`
- Artifact score: `0.014`
- Labels: `inspection_candidate`

Example:
```text
activation=533.56, source=pile-10k, text_id=272, token_pos=253
. Domestic violence can begin mildly, through controlling behavior, but often escalates to physical trauma or even[ death].


```

## Suspicious coactivation pairs

### Pair 5681 / 9073
- Labels: `likely_artifact_pair`
- Pair suspicion score: `1.000`
- Shared token concentration: `1.000`
- Jaccard: `0.6027397260273972`
- PMI: `6.769004541143627`
- Coactivation count: `44`

Shared top tokens:
```text
{
  "\n\n": 88
}
```

### Pair 235 / 14678
- Labels: `likely_artifact_pair`
- Pair suspicion score: `1.000`
- Shared token concentration: `1.000`
- Jaccard: `0.5789473684210527`
- PMI: `6.723884105863157`
- Coactivation count: `44`

Shared top tokens:
```text
{
  "Q": 88
}
```

### Pair 286 / 431
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.813`
- Shared token concentration: `0.813`
- Jaccard: `0.971830985915493`
- PMI: `6.847561916200972`
- Coactivation count: `69`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  ".": 10,
  " ": 8,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 4 / 286
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.813`
- Shared token concentration: `0.813`
- Jaccard: `0.9583333333333334`
- PMI: `6.833377281209015`
- Coactivation count: `69`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  ".": 10,
  " ": 8,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 132 / 286
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.813`
- Shared token concentration: `0.813`
- Jaccard: `0.9583333333333334`
- PMI: `6.833377281209015`
- Coactivation count: `69`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  ".": 10,
  " ": 8,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 143 / 286
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.813`
- Shared token concentration: `0.813`
- Jaccard: `0.9583333333333334`
- PMI: `6.833377281209015`
- Coactivation count: `69`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  ".": 10,
  " ": 8,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 286 / 391
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.813`
- Shared token concentration: `0.813`
- Jaccard: `0.9583333333333334`
- PMI: `6.833377281209015`
- Coactivation count: `69`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  ".": 10,
  " ": 8,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 286 / 502
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.813`
- Shared token concentration: `0.813`
- Jaccard: `0.92`
- PMI: `6.791992065046161`
- Coactivation count: `69`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  ".": 10,
  " ": 8,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 4 / 431
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.811`
- Shared token concentration: `0.811`
- Jaccard: `0.9859154929577465`
- PMI: `6.8477660186611145`
- Coactivation count: `70`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  " ": 10,
  ".": 10,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 132 / 431
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.811`
- Shared token concentration: `0.811`
- Jaccard: `0.9859154929577465`
- PMI: `6.8477660186611145`
- Coactivation count: `70`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  " ": 10,
  ".": 10,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 143 / 431
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.811`
- Shared token concentration: `0.811`
- Jaccard: `0.9859154929577465`
- PMI: `6.8477660186611145`
- Coactivation count: `70`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  " ": 10,
  ".": 10,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 391 / 431
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.811`
- Shared token concentration: `0.811`
- Jaccard: `0.9859154929577465`
- PMI: `6.8477660186611145`
- Coactivation count: `70`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  " ": 10,
  ".": 10,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 4 / 132
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.811`
- Shared token concentration: `0.811`
- Jaccard: `0.9722222222222222`
- PMI: `6.833581383669158`
- Coactivation count: `70`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  " ": 10,
  ".": 10,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 4 / 143
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.811`
- Shared token concentration: `0.811`
- Jaccard: `0.9722222222222222`
- PMI: `6.833581383669158`
- Coactivation count: `70`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  " ": 10,
  ".": 10,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 4 / 391
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.811`
- Shared token concentration: `0.811`
- Jaccard: `0.9722222222222222`
- PMI: `6.833581383669158`
- Coactivation count: `70`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  " ": 10,
  ".": 10,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 132 / 143
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.811`
- Shared token concentration: `0.811`
- Jaccard: `0.9722222222222222`
- PMI: `6.833581383669158`
- Coactivation count: `70`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  " ": 10,
  ".": 10,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 132 / 391
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.811`
- Shared token concentration: `0.811`
- Jaccard: `0.9722222222222222`
- PMI: `6.833581383669158`
- Coactivation count: `70`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  " ": 10,
  ".": 10,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 143 / 391
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.811`
- Shared token concentration: `0.811`
- Jaccard: `0.9722222222222222`
- PMI: `6.833581383669158`
- Coactivation count: `70`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  " ": 10,
  ".": 10,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 431 / 502
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.811`
- Shared token concentration: `0.811`
- Jaccard: `0.9459459459459459`
- PMI: `6.806380802498261`
- Coactivation count: `70`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  " ": 10,
  ".": 10,
  "/": 4,
  "1": 4,
  "0": 4
}
```

### Pair 4 / 502
- Labels: `likely_artifact_pair`
- Pair suspicion score: `0.811`
- Shared token concentration: `0.811`
- Jaccard: `0.9333333333333333`
- PMI: `6.792196167506304`
- Coactivation count: `70`

Shared top tokens:
```text
{
  "<bos>": 600,
  "'": 60,
  "’": 48,
  " ": 10,
  ".": 10,
  "/": 4,
  "1": 4,
  "0": 4
}
```

## Bimodality candidate summaries

### Feature 179
- Labels: `none`
- Artifact score: `0.126`
- Semantic score: `0.783`

### Feature 227
- Labels: `none`
- Artifact score: `0.099`
- Semantic score: `0.835`

### Feature 98
- Labels: `none`
- Artifact score: `0.054`
- Semantic score: `0.917`

### Feature 507
- Labels: `inspection_candidate`
- Artifact score: `0.034`
- Semantic score: `0.953`

### Feature 115
- Labels: `none`
- Artifact score: `0.072`
- Semantic score: `0.885`

### Feature 347
- Labels: `none`
- Artifact score: `0.041`
- Semantic score: `0.937`

### Feature 55
- Labels: `none`
- Artifact score: `0.058`
- Semantic score: `0.907`

### Feature 446
- Labels: `none`
- Artifact score: `0.053`
- Semantic score: `0.915`

### Feature 463
- Labels: `none`
- Artifact score: `0.150`
- Semantic score: `0.745`

### Feature 449
- Labels: `none`
- Artifact score: `0.077`
- Semantic score: `0.863`

### Feature 118
- Labels: `none`
- Artifact score: `0.046`
- Semantic score: `0.927`

### Feature 353
- Labels: `none`
- Artifact score: `0.059`
- Semantic score: `0.910`

### Feature 475
- Labels: `none`
- Artifact score: `0.068`
- Semantic score: `0.882`

### Feature 2496
- Labels: `none`
- Artifact score: `0.067`
- Semantic score: `0.890`

### Feature 9835
- Labels: `none`
- Artifact score: `0.089`
- Semantic score: `0.849`

### Feature 4006
- Labels: `none`
- Artifact score: `0.144`
- Semantic score: `0.744`

### Feature 14588
- Labels: `none`
- Artifact score: `0.045`
- Semantic score: `0.921`

### Feature 14110
- Labels: `none`
- Artifact score: `0.140`
- Semantic score: `0.761`

### Feature 12227
- Labels: `none`
- Artifact score: `0.170`
- Semantic score: `0.719`

### Feature 16331
- Labels: `none`
- Artifact score: `0.095`
- Semantic score: `0.836`

### Feature 14862
- Labels: `none`
- Artifact score: `0.132`
- Semantic score: `0.766`

### Feature 14832
- Labels: `none`
- Artifact score: `0.079`
- Semantic score: `0.872`

### Feature 134
- Labels: `none`
- Artifact score: `0.087`
- Semantic score: `0.856`

### Feature 14358
- Labels: `none`
- Artifact score: `0.128`
- Semantic score: `0.785`

### Feature 13683
- Labels: `none`
- Artifact score: `0.097`
- Semantic score: `0.843`

### Feature 14365
- Labels: `punctuation_like`
- Artifact score: `0.176`
- Semantic score: `0.694`

### Feature 15345
- Labels: `none`
- Artifact score: `0.036`
- Semantic score: `0.938`

### Feature 206
- Labels: `punctuation_like, space_like`
- Artifact score: `0.270`
- Semantic score: `0.566`

### Feature 8629
- Labels: `none`
- Artifact score: `0.123`
- Semantic score: `0.799`

### Feature 9758
- Labels: `none`
- Artifact score: `0.087`
- Semantic score: `0.863`
