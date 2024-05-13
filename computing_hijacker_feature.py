import editdistance
from BML.transform import BaseTransform
import sys
sys.path.append('/home/ycxie/xyc/MOAS/venv/computing_hijacker_feature.py')

# Extracting relevant features of hijacker ASN from BGP data stream
class Hijacker_features(BaseTransform):

    computeRoutes = False

    def __init__(self, primingFile, dataFile, params, outFolder, logFiles):

        BaseTransform.__init__(self, primingFile, dataFile, params, outFolder, logFiles)
        self.asn = outFolder.split('/')[2].split('_')[-1]
        print(self.asn)

    def transforms(self, index, routes, updates):
        features = {
            "nb_A_hijacker": 0,  # Number of hijacker announcements
            "nb_implicit_W_hijacker": 0, # Number of hijacker implicit withdrawals
            "nb_dup_A_hijacker": 0, # Number of hijacker duplicate announcements
            "nb_A_prefix_hijacker": 0,  # Number of hijacker announced prefixes
            "max_A_prefix_hijacker": 0,  # Max. announcements the prefix of hijacker
            "avg_A_prefix_hijacker": 0,  # Avg. announcements the prefix of hijacker
            "nb_orign_change_hijacker": 0, # ORIGIN attribute (IGP, EGP, INCOMPLETE), which was announced previously, changed with a new reannouncement of hijancker.
            "nb_new_A_hijacker": 0,  # Not stored in RIB
            "nb_new_A_afterW_hijacker": 0,
            "max_path_len_hijacker": 0,
            "avg_path_len_hijacker": 0,
            "max_editdist_hijacker": 0,
            "avg_editdist_hijacker": 0,
            "editdist_7_hijacker": 0,
            "editdist_8_hijacker": 0,
            "editdist_9_hijacker": 0,
            "editdist_10_hijacker": 0,
            "editdist_11_hijacker": 0,
            "editdist_12_hijacker": 0,
            "editdist_13_hijacker": 0,
            "editdist_14_hijacker": 0,
            "editdist_15_hijacker": 0,
            "editdist_16_hijacker": 0,
            "editdist_17_hijacker": 0,
            "nb_tolonger_hijacker": 0,
            "nb_toshorter_hijacker": 0,
            "avg_interarrival_hijacker": 0, # The time interval for hijackers to update messages
        }

        if (len(updates) < 2):
            return (features)

        prevTime = 0

        A_hijacker_prefix = {}

        path_len = []
        inter_time = []
        editDist = []

        for update in updates:

            if(update["fields"]["prefix"] not in routes):
                routes[update["fields"]["prefix"]] = {}
            if(update["collector"] not in routes[update["fields"]["prefix"]]):
                routes[update["fields"]["prefix"]][update["collector"]] = {}

            if(update["type"]=='A') and (self.asn == update["fields"]["as-path"].split(" ")[-1]):

                features["nb_A_hijacker"] += 1

                Uas_path = update["fields"]['as-path']

                path_len.append(len(Uas_path.split(" ")))

                if(update["fields"]["prefix"] not in A_hijacker_prefix):
                    A_hijacker_prefix[update["fields"]["prefix"]] = 0

                A_hijacker_prefix[update["fields"]["prefix"]] += 1

                if (update["peer_asn"] not in routes[update["fields"]["prefix"]][update["collector"]] or
                        routes[update["fields"]["prefix"]][update["collector"]][update["peer_asn"]] == None):
                    features["nb_new_A_hijacker"] += 1
                elif (routes[update["fields"]["prefix"]][update["collector"]][update["peer_asn"]] == "w" + str(index)):
                    features["nb_new_A_afterW_hijacker"] += 1
                elif (routes[update["fields"]["prefix"]][update["collector"]][update["peer_asn"]][0] == "w"):
                    features["nb_new_A_hijacker"] += 1
                elif (routes[update["fields"]["prefix"]][update["collector"]][update["peer_asn"]] == Uas_path):
                    features["nb_dup_A_hijacker"] += 1
                else:
                    features["nb_implicit_W_hijacker"] += 1

                    ASlist_prev = routes[update["fields"]["prefix"]][update["collector"]][update["peer_asn"]].split(" ")
                    ASlist_new = Uas_path.split(" ")

                    if (ASlist_prev[-1] != ASlist_new[-1]):
                        features["nb_orign_change_hijacker"] += 1

                    if (len(ASlist_new) > len(ASlist_prev)):
                        features["nb_tolonger_hijacker"] += 1
                    else:
                        features["nb_toshorter_hijacker"] += 1

                    edist = editdistance.eval(ASlist_prev, ASlist_new)
                    editDist.append(edist)
                    if (edist >= 7 and edist <= 17):
                        features["editdist_" + str(edist) + '_hijacker'] += 1

                routes[update["fields"]["prefix"]][update["collector"]][update["peer_asn"]] = Uas_path

            if (prevTime != 0):
                iTime = int(update["time"]) - prevTime
                if (iTime > 0):
                    inter_time.append(iTime)

            prevTime = int(update["time"])


        features["nb_A_prefix_hijacker"] = len(A_hijacker_prefix)

        A_hijacker_prefix_values = A_hijacker_prefix.values()
        if (len(A_hijacker_prefix_values) > 1):
            features["max_A_prefix_hijacker"] = max(A_hijacker_prefix_values)
            features["avg_A_prefix_hijacker"] = round(sum(A_hijacker_prefix_values) / len(A_hijacker_prefix_values))

        if (len(path_len) > 1):
            features["max_path_len_hijacker"] = max(path_len)
            features["avg_path_len_hijacker"] = round(sum(path_len) / len(path_len))

        if (len(inter_time) > 1):
            features["avg_interarrival_hijacker"] = round(sum(inter_time) * 1000 / len(inter_time))

        if (len(editDist) > 1):
            features["max_editdist_hijacker"] = max(editDist)
            features["avg_editdist_hijacker"] = round(sum(editDist) / len(editDist))

        return (features)

    def postProcess(self, transformedData):
        transformedData.pop(0)
        return(transformedData)