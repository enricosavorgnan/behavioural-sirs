# Tesi
## Modello SIRS - Main stream



## Indice
- [Introduzione](#introduzione)
- [Workflow](#workflow)
- [Modello SIRS base](#modello-sirs-base)
- [Folders](#folders)
  - [Libri](#libri)
  - [Photo](#photo)

## Introduzione

Ho inizialmente considerato il modello semplice SIRS:
$$
\begin{cases}
\dot{S} = \mu - \mu S - \beta S I + \theta R \\
\dot{I} = \beta S I - I (\gamma + \mu) \\
\dot{R} = \gamma I - R (\theta + \mu) 
\end{cases}
$$

## 
Questo sistema può essere riscritto considerando le sole equazioni S ed I:
$$
\begin{cases}
\dot{S} = \mu - \mu S - \beta S I + \theta (1 - S - I) \\
\dot{I} = \beta S I - I (\gamma + \mu)
\end{cases}
$$

che, riscrivendo la prima equazione, diventa:
$$
\begin{cases}
\dot{S} = (\mu + \theta) - S (\mu + \theta) - \beta S I - \theta I \\
\dot{I} = \beta S I - I (\gamma + \mu)
\end{cases}
$$

  
##
Oppure, ed è la formulazione su cui mi sono inizialmente soffermato, considerando le sole equazione I ed R:
$$
\begin{cases}
\dot{I} = \beta S I - I (\gamma + \mu) \\
\dot{R} = \gamma I - R (\theta + \mu)
\end{cases}
$$
  
Anche in questo caso la prima equazione può essere riscritta come:
$$
\begin{cases}
\dot{I} = \beta I (1 - R - I) - I (\gamma + \mu) \\
\dot{R} = \gamma I - R (\theta + \mu)
\end{cases}
$$
da cui si ottiene con semplici passaggi:
$$
\begin{cases}
\dot{I} = I (\beta - (\gamma + \mu)) - \beta RI - \beta I^{2} \\
\dot{R} = \gamma I - R (\theta + \mu)
\end{cases}
$$


##
##

## Workflow
In questa sezione verranno riportati i vari step del lavoro, ciascuno con una descrizione e un'eventuale spiegazione.

-  **Modello SIRS base**
    1. Ricavo dei punti di equilibrio
    2. Studio per $t$ piccolo 
       1. Studio Desease Free Equilibrium (DFE)
       2. Studio delle biforcazioni
   3. Studio per $t$ grande
      1. Studio DFE
      2. Studio EE
   4. Analisi delle biforcazioni di Hopf
   
[//]: # (TODO: aggiungere workflow per i modelli successivi)
- **Modello SIRS con *seasonality***: $\beta(t)$ periodica
- **Modello SIRS con *fading memory***
- **Modello SIRS con *fading memory*e *vaccination***
- **Modello SIRS con *fading memory*e *vaccination***
- **Modello SIRS con *fading memory*, *vaccination* e *quarantine***


<br><br><br>

***

## Modello SIRS base
### 1. Ricavo dei punti di equilibrio
I punti di equilibrio del sistema sono dati da:
$$
\text{DFE:}
\begin{cases}
\dot{S} = 1 \\
\dot{I} = 0 \\
\dot{R} = 0
\end{cases}
\quad \land \quad
\text{EE:}
\begin{cases}
\dot{S} = \frac{1}{R_{0}} \\
\dot{I} = (1 - \frac{1}{R_{0}})(\frac{\mu + \theta}{\mu + \gamma + \theta}) \\
\dot{R} = (1 - \frac{1}{R_{0}})(\frac{\gamma}{\mu + \gamma + \theta})
\end{cases}
$$

##
In particolare, il secondo sistema di equazioni è ottenuto risolvendo il seguente sistema:
$$
\begin{cases}
I(\beta - (\gamma + \mu)) = \beta RI - \beta I^{2} \\
\gamma I = R (\mu + \theta) \\
\end{cases}
$$

### 2. Studio per $t$ piccolo
Per $t$ piccolo si ha che $\mu \approx 0$. Inoltre, si ha che la frazione di infetti $I$ è molto piccola, pari ad $\epsilon$, la frazione di rimossi $R$ è pari a 0 e che quindi il numero di suscettibili $S$ è pari a $1-\epsilon$.
Di conseguenza, possiamo riscrivere il sistema come:
$$
\begin{cases}
\dot{I} = I (\beta (1-R) - \gamma) - \beta I^{2} \approx I(\beta - \gamma) - \beta I^{2} \\
\dot{R} = \gamma I - \theta R
\end{cases}
$$

Applicando il comodo teoremino dei due piccioni con una fava, si può maggiorare la prima equazione rimuovendo il termine quadratico, ottenendo un sistema di equazioni per $Z(t) = I(t)$ con soluzione (sempre approssimata), pari a $Z(t) = I(0) e^{(\beta - \gamma)t}$.

Questo risultato permette di dire che, per $\frac{\beta}{\gamma} < 1$, la variabile $I$ tende a 0, mentre per $\frac{\beta}{\gamma} > 1$ la variabile $I$ tende a $\infty$.  
Da notare che per $\frac{\beta}{\gamma} = 1$ si ha un punto di biforcazione.

[//]: # (TODO: aggiungere studio delle biforcazioni)
#### Studio delle biforcazioni
Per $\frac{\beta}{\gamma} = 1$ si ha un punto di biforcazione, che si può studiare.

!!! HA SENSO SE RAGIONIAMO PER $t$ piccolo?


<br> <br> <br>

***



## Folders 
[//]: # (TODO: aggiornare cartelle)
### libri
- `J85 Physics Reports 2016.pdf` : Leggi tutto. È un articolo riguardo a $???$.  
- `Keeling & Rohani - Modeling Infectious Diseases in Humans and Animals.pdf` : Leggi le parti in azzurro. È un libro riguardo a $???$.
-

### photo
`presentation_1.jpg` : prima slide. 
Distingue in due colonne:
- il modello "base", con $0 < t << k$, quindi $\mu \approx 0$
- il modello "realistico", con $t >> k$, e anche con $\mu$.  

Da notare in generale la presenza del parametro $\theta$, che indica la frazione di $R$ che perde l'immunità, e quindi contribuisce all'aumento della popolazione $P$.


`presentation_2.jpg` : seconda slide. Si nota in particolare la riduzione a due equazioni, sempre nei due casi "base" (sx) e "realistico" (dx).
Il parametro M rappresenta la popolazione $???$.



`presentation_3.jpg` : boh random.


`presentation_1b.jpg`, `presentation_1c.jpg` e `presentation_2b.jpg`  : stesse foto ma con ERRORE: include anche nel modello "base" il parametro $\mu$.

