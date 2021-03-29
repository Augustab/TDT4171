'''Oppgave 1 - Hidden Markov Model'''

__author__ = "August Asheim Birkeland"

import numpy as np


class HiddenMarkovModel:
    """class takes inn numpy-arrays for prior probability, transition probability, evidence probability, evidence
    and the evidence-matrixes O (where O[0] represents the matrix for evidene=true and omvendt."""
    def __init__(self, prior_p, yesterday_today_p, evidence_p, evidence, O):
        self.prior_p = prior_p
        self.yesterday_today_p = yesterday_today_p
        self.evidence_p = evidence_p
        self.evidence = evidence
        self.O = O

    def forward(self, t):
        """Forward algoritmen basert på matrise-teorien i boka, nærmere bestemt side 579,
        ligning (15.12): f_1_t+1 = alfa*O_t*transposed(T)*f_1_t"""
        # I første iterasjon er current_P = prior probability.
        current_P = self.prior_p
        for i in range(t):
            # Denne sjekken sørger for at koden ikke leter etter evidence man ikke har.
            if i < len(self.evidence):
                # Merk at jeg bruker np.dot for å beregne dot-product mellom to matriser.
                current_P = np.dot(np.dot(self.O[self.evidence[i]], np.transpose(self.yesterday_today_p)), current_P)
                current_P = current_P * (1 / sum(current_P))
            # Dersom vi skal PREDICTE går programmet inn i denne elsen.
            else:
                # Her er det ikke noe evidence i utregningen, kun transition matrise og current_P-resultat.
                current_P = np.dot(np.transpose(self.yesterday_today_p), current_P)
        return current_P

    def backward(self, b_t_t, k, t):
        """Backward algoritmen basert på matrise-teorien i boka, nærmere bestemt side 579,
        ligning (15.13): b_k+1_t = T*O_k+1*B_k+2"""
        # I første iterasjon vil current_P være void = [1, 1]
        current_P = b_t_t
        # Itererer fra høyeste t-verdi og nedover siden funksjonen bruker høyere b-verdier i utregningen.
        for i in range(t-1, k-1, -1):
            current_P = np.dot(np.dot(self.yesterday_today_p, self.O[self.evidence[i]]), current_P)
            # Merk at man ikke trenger å normalisere når man kjører backward-algoritmen, men man kan om man vil
            # current_P = current_P*(1/sum(current_P))
        return current_P

    def forward_backward(self, k, t):
        """Smoothing-algoritmen som tar i bruk både forward og backward.
        Basert på teorien på siden https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
        P(X_k|all evidence) = F_k*B_k+1_t"""
        f_k = self.forward(k)
        #sender inn void-matrisen [1,1]
        b_k_1 = self.backward(np.array([1, 1]), k, t)
        #print(f"Forward: {f_k} * Backward: {b_k_1}")
        result = f_k*b_k_1
        return result*(1/sum(result))

    def viterbi(self, evidence):
        """Most Likely Explanation koden, også kalt Viterbi.
        Jeg har basert meg i stor grad på umbrella-problemet og gjeldende powerpoint
        """
        rows = self.yesterday_today_p.shape[0]
        columns = len(evidence)
        path = [0]*columns
        # Dette er matrisen jeg bruker for å lagre Max-verdiene etterhvert som jeg jobber meg fremover
        M = np.zeros((rows, columns))
        # Her initialiserer jeg den første kolonnen, som vil være lik forward(1)
        init = np.dot(np.dot(self.O[self.evidence[0]], np.transpose(self.yesterday_today_p)), self.prior_p)
        M[:, 0] = init*(1/sum(init))
        # Linjen under setter første verdi på stien til true eller false avhengig av hvilken som er mest sannsynlig
        path[0] = True if M[0, 0] > M[1, 0] else False

        for t in range(1, columns):
            # n vil være verdien til X_t
            for n in range(rows):
                #print(f"{M[:, t - 1]},{np.transpose(self.yesterday_today_p)[:, n]},{self.evidence_p[evidence[t]][n]}")
                M_n_t = M[:, t - 1] * np.transpose(self.yesterday_today_p)[:, n] * self.evidence_p[evidence[t]][n]
                M[n, t] = np.max(M_n_t)
            # Linjen under setter verdi t på stien til true eller false avhengig av hvilken som er mest sannsynlig
            path[t] = True if M[0, t] > M[1, t] else False
        return M, path


def problem1():
    """Initialiserer problemet med HMM-klassen min"""
    prior_p = np.array([0.5, 0.5])
    yesterday_today_p = np.array([[0.8, 0.2],
                                  [0.3, 0.7]])
    evidence_p = np.array([[0.75, 0.2],
                           [0.25, 0.8]])
    #MERK: jeg bruker 0=true og 1=false
    evidence = [0, 0, 1, 0, 1, 0]
    #Disse O-matrisene blir brukt avhengig av om evidence er true eller false.
    O = [np.array([[0.75, 0], [0, 0.2]]), np.array([[0.25, 0], [0, 0.8]])]
    hmm = HiddenMarkovModel(prior_p, yesterday_today_p, evidence_p, evidence, O)

    #Part b) FILTERING
    print("------ Part b) FILTERING ------")
    for i in range(1, 7):
        print(f"P(X_{i}|e_1:{i}) = {hmm.forward(i)}")
    print("")
    #Part c) PREDICTION
    print("------ Part c) PREDICTION ------")
    for i in range(7,31):
        print(f"P(X_{i}|e_1:6) = {hmm.forward(i)}")
    print("")
    #PART d) SMOOTHING!!
    print("------ Part d) SMOOTHING ------")
    for i in range(0,6):
        print(f"P(X_{i}|e_1:6) = {hmm.forward_backward(i, 6)}")
    print("")
    #PART e) Most Likely Explanation MLE!!
    print("------ Part e) MOST LIKELY EXPLANATION ------")
    V, prev = hmm.viterbi(evidence)
    print("The probabilities for Fish (row 0) vs Not Fish (row 1)")
    for i in range(V.shape[1]):
        print(f"ArgMax X_{i}=true = {round(V[0,i],7)} | ArgMax X_{i}=false = {round(V[1,i],7)}")
    print(V)
    print("The resulting Most Likely Explanation Sequence, FISH:", prev)


def problem_umbrella():
    prior_p = np.array([0.5, 0.5])
    yesterday_today_p = np.array([[0.7, 0.3], [0.3, 0.7]])
    evidence_p = np.array([[0.9, 0.2], [0.1, 0.8]])
    O = [np.array([[0.9, 0], [0, 0.2]]), np.array([[0.1, 0], [0, 0.8]])]
    evidence = [0, 0, 1, 0, 0]
    hmm = HiddenMarkovModel(prior_p, yesterday_today_p, evidence_p, evidence, O)


if __name__ == '__main__':
    problem1()
    #problem_umbrella()