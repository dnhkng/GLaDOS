// Console language. The markup and scripts stay in English (as upstream ships them); this layer
// swaps exact English strings for their translation as they are rendered, and back again.
(() => {
  const FR = {
    // Navigation & header
    'Theme': 'Thème', 'Interface theme': "Thème de l'interface", 'Language': 'Langue',
    'Presence screen ↗': 'Écran de présence ↗', 'Full view': 'Vue complète', 'Display': 'Affichage', 'Simple view': 'Vue simple', 'CONNECTING': 'CONNEXION', 'Views': 'Vues',
    'Enrichment Center': 'Accueil', 'Cores': 'Noyaux', 'Test Chamber': 'Mémoire',
    'Neural Buffer': 'Contexte', 'Facility Tools': 'Outils', 'Test Assignments': 'Tâches',
    'Facility Settings': 'Réglages', 'Observation Log': 'Journal', 'Diagnostics': 'Diagnostics',
    'LIVE TELEMETRY LINKED': 'TÉLÉMÉTRIE EN DIRECT', 'LIVE TELEMETRY': 'TÉLÉMÉTRIE EN DIRECT',
    'DISCONNECTED · RETRYING': 'DÉCONNECTÉ · NOUVELLE TENTATIVE', 'DEMO DATA': 'DONNÉES DE DÉMO',
    // Home
    'Aperture Science Enrichment Center. Inspect each core’s activity, controls and reports.':
      "Centre d'enrichissement Aperture Science. Consultez l'activité, les commandes et les rapports de chaque noyau.",
    'Use this device’s microphone': 'Utiliser le micro de cet appareil',
    'Facility observation and control.': 'Observation et contrôle du centre.',
    'GLaDOS eye and optic feed': 'Œil de GLaDOS et flux optique', 'GLaDOS aperture, connecting': 'Œil de GLaDOS, connexion',
    'Connecting': 'Connexion', 'Connecting…': 'Connexion…', 'You': 'Vous', 'No input received yet.': 'Aucune entrée reçue pour le moment.',
    'Replies appear here, including when voice output is muted.': 'Les réponses apparaissent ici, même quand la voix est coupée.',
    'Live optic feed with theme treatment': 'Flux optique en direct avec effet de thème',
    'TEST SUBJECT': 'SUJET DE TEST', 'SUBJECT': 'SUJET', 'MOVEMENT': 'MOUVEMENT', 'MOTION': 'MOUVEMENT',
    'No observation yet.': 'Aucune observation pour le moment.', 'On her mind': 'Ce qui l’occupe',
    'Talk to GLaDOS': 'Parler à GLaDOS', 'Send a message or ask her to work on a task': 'Envoyez un message ou confiez-lui une tâche',
    'What would you like to do?': 'Que voulez-vous faire ?', 'Send message': 'Envoyer',
    'Latest turn': 'Dernier échange', 'Waiting for input': "En attente d'une entrée",
    'Recorded server events: input accepted → playback requested. Excludes endpoint silence and browser audio buffering.':
      'Événements serveur enregistrés : entrée acceptée → lecture demandée. Hors silence de fin de phrase et tampon audio du navigateur.',
    'Timing begins with the next user input.': "Le chronométrage commence à la prochaine entrée.",
    'Telemetry disconnected': 'Télémétrie déconnectée', 'Turn in progress': 'Échange en cours',
    'Routing completed': 'Routage terminé', 'Reply inference admitted': 'Inférence de réponse admise',
    'First response token': 'Premier jeton de réponse', 'First clause to TTS': 'Première phrase vers la synthèse',
    'First audio ready': 'Premier audio prêt', 'Playback requested': 'Lecture demandée',
    'Inference slots': "Emplacements d'inférence", 'Inspect': 'Inspecter', 'Core activity': 'Activité des noyaux',
    'Core': 'Noyau', 'State': 'État', 'Latest update': 'Dernière mise à jour', 'Settings': 'Réglages',
    'Assignments tracked →': 'Missions suivies →', 'Online cores →': 'Noyaux en ligne →',
    'User-facing inferences →': 'Inférences utilisateur →', 'Dual Lane Orchestration': 'Orchestration à deux voies',
    'Priority Lane': 'Voie prioritaire', 'routing and conversation requests': 'routage et conversation',
    'Background Lane': "Voie d'arrière-plan", 'emotion and background cores': 'émotion et noyaux de fond',
    'Engine state': 'État du moteur', 'Microphone': 'Microphone', 'Voice output': 'Voix', 'Camera': 'Caméra',
    'Quiet': 'Silence', 'Pleasure': 'Plaisir', 'Arousal': 'Excitation', 'Dominance': 'Dominance',
    'Edit response instructions': 'Modifier les consignes de réponse', 'Inspect context': 'Inspecter le contexte',
    'Inspect emotion': "Inspecter l'émotion", 'Last user': 'Dernier message', 'Last spoke': 'Dernière prise de parole',
    'Since last user': 'Depuis votre dernier message', 'Since last spoke': 'Depuis sa dernière parole',
    'Test assignments': 'Tâches', 'Create assignment': 'Créer une mission', 'Edit assignment': 'Modifier la mission',
    'Assignments are shared with the Central Core. Ask GLaDOS to work on an open assignment.':
      'Les missions sont partagées avec le noyau central. Demandez à GLaDOS de travailler sur une mission ouverte.',
    'Observation log · live stream': "Journal d'observation · direct", 'open inspector →': "ouvrir l'inspecteur →",
    'Ready when you are.': 'Prête quand vous voulez.', 'Waiting for GLaDOS…': 'En attente de GLaDOS…',
    'Enter a message.': 'Saisissez un message.', 'Message sent to GLaDOS.': 'Message envoyé à GLaDOS.',
    // Cores
    'Central Core': 'Noyau central', 'Central Core · GLaDOS': 'Noyau central · GLaDOS', 'Autonomy Core': "Noyau d'autonomie",
    'Vision Core': 'Noyau Vision', 'Emotion Core': 'Noyau Émotion', 'Memory Core': 'Noyau Mémoire', 'Health Core': 'Noyau Santé',
    'Search Core': 'Noyau Recherche', 'Routing Core': 'Noyau Routage', 'Nexus Core': 'Noyau Nexus', 'Weather Core': 'Noyau Météo',
    'Planning Core': 'Noyau Planification', 'Maintenance Core': 'Noyau Maintenance',
    'Dedicated functions. Shared processing.': 'Fonctions dédiées. Calcul partagé.', 'system diagnostics →': 'diagnostics système →',
    'The Central Core handles your requests. Other cores maintain their own functions, state and reports. Suspend a core to stop future scheduled runs; a current run may finish.':
      "Le noyau central traite vos demandes. Les autres noyaux gèrent leurs propres fonctions, état et rapports. Suspendre un noyau arrête ses prochaines exécutions ; celle en cours peut se terminer.",
    'Suspended': 'Suspendu', 'Waiting for resume': 'En attente de reprise', 'Working': 'En cours', 'Queued': 'En file',
    'Waiting for user response': 'En attente de votre réponse', 'Waiting for a mind worker': "En attente d'un exécuteur",
    'Processing update': 'Traitement de la mise à jour', 'Offline': 'Hors ligne', 'Function disabled': 'Fonction désactivée',
    'Standby': 'En veille', 'On demand': 'À la demande', 'Ready for requests': 'Prêt pour les demandes',
    'Awaiting first report.': 'En attente du premier rapport.', 'Important · review': 'Important · à examiner',
    'Regular update': 'Mise à jour courante', 'Cancel task': 'Annuler la tâche', 'Cancellation requested': 'Annulation demandée',
    'Resume core': 'Reprendre le noyau', 'Suspend core': 'Suspendre le noyau', 'No output yet.': 'Aucune sortie pour le moment.',
    'Schedule updated.': 'Planification mise à jour.', 'One run queued. The result will refresh when it finishes.':
      'Une exécution en file. Le résultat sera actualisé une fois terminée.',
    'On requested research': 'Sur demande de recherche', 'Completed/attempted runs': 'Exécutions réussies/tentées',
    'Research topic': 'Sujet de recherche', 'Waiting for a request': "En attente d'une demande", 'Search rounds': 'Tours de recherche',
    'Cited passages': 'Passages cités', 'Time limit': 'Limite de temps', 'Recent messages kept': 'Messages récents conservés',
    'Raw messages': 'Messages bruts', 'Estimated context': 'Contexte estimé', 'Recall topic': 'Sujet de rappel',
    'Waiting for a topic': "En attente d'un sujet", 'Autonomy Core unavailable.': "Noyau d'autonomie indisponible.",
    'No pending handoff to Central Core.': 'Aucun transfert en attente vers le noyau central.',
    'PAD schedule paused': 'Planification PAD en pause', 'PAD agent active': 'Agent PAD actif',
    'Emotion tracking is disabled in this profile. Enable autonomy.emotion.enabled to use the PAD gauges.':
      "Le suivi des émotions est désactivé dans ce profil. Activez autonomy.emotion.enabled pour utiliser les jauges PAD.",
    // Processing / tools
    'Shared processing · active requests and queues': 'Calcul partagé · demandes actives et files',
    'Cores share the available processing channels. Routing and your replies have priority; background cores use spare capacity.':
      'Les noyaux partagent les canaux de calcul. Le routage et vos réponses sont prioritaires ; les noyaux de fond utilisent la capacité restante.',
    'Facility command queue': 'File de commandes du centre', 'Run a task': 'Lancer une tâche',
    'Fixed, read-only host operations run on the CPU alongside the cores.': "Opérations fixes en lecture seule, exécutées sur le CPU à côté des noyaux.",
    'Waiting for capacity': 'En attente de capacité', 'None': 'Aucun', 'Latest routing decision': 'Dernière décision de routage',
    'No decision yet.': 'Aucune décision pour le moment.', 'No routing decision yet.': 'Aucune décision de routage pour le moment.',
    'capabilities available to GLaDOS right now.': 'capacités disponibles pour GLaDOS en ce moment.',
    'Tools run when called by a core. Inspect their inputs or run a fixed host command here. Local time and date are already supplied in context.':
      "Les outils s'exécutent à l'appel d'un noyau. Inspectez leurs entrées ou lancez ici une commande fixe. L'heure et la date locales sont déjà fournies dans le contexte.",
    'No callable tools reported. Connect to the engine to inspect capabilities.': 'Aucun outil signalé. Connectez le moteur pour inspecter les capacités.',
    'No description': 'Aucune description', 'Inspect tool': "Inspecter l'outil", 'Completed': 'Terminé', 'Failed': 'Échec',
    'No output': 'Aucune sortie', 'Command execution': 'Exécution de commande', 'Facility command queue unavailable.': 'File de commandes indisponible.',
    'Capacity unavailable': 'Capacité indisponible', 'Capacity unavailable.': 'Capacité indisponible.', 'Active': 'Actif', 'Idle': 'Inactif',
    'Reserved for interactive requests': 'Réservé aux demandes interactives', 'Shared': 'Partagé', 'Single step': 'Étape unique',
    'Preview only': 'Aperçu seulement', 'Selected action': 'Action choisie', 'Action': 'Action', 'Accepted': 'Accepté',
    'Full assistant with read-only tools': 'Assistant complet avec outils en lecture seule', 'Fallback used': 'Repli utilisé',
    'MCP server': 'Serveur MCP', 'Selected tools': 'Outils choisis', 'OPTION PREFERENCE': 'PRÉFÉRENCE DES OPTIONS',
    'In-flight requests': 'Demandes en cours', 'Available capacity': 'Capacité disponible', 'Queued requests': 'Demandes en file',
    'Interactive reservation': 'Réservation interactive', 'Shared capacity': 'Capacité partagée',
    'User interaction active · new background inference paused. Running requests may finish.':
      "Interaction en cours · nouvelles inférences de fond en pause. Les demandes en cours peuvent se terminer.",
    'No requests waiting.': 'Aucune demande en attente.', 'No update yet.': 'Aucune mise à jour pour le moment.',
    // Tasks
    'saved work and results.': 'travaux et résultats enregistrés.',
    'Create an assignment, ask GLaDOS to work on it, then inspect its result. Core state and reports live in Cores.':
      "Créez une mission, demandez à GLaDOS d'y travailler, puis consultez le résultat. L'état et les rapports des noyaux sont dans Noyaux.",
    'No summary': 'Aucun résumé', 'No report yet.': 'Aucun rapport pour le moment.', 'Reopen': 'Rouvrir', 'Mark done': 'Marquer terminé',
    'Sent to GLaDOS. The task updates when she saves a result.': 'Envoyé à GLaDOS. La tâche sera mise à jour quand elle enregistrera un résultat.',
    'Task updated.': 'Tâche mise à jour.', 'Task unavailable': 'Tâche indisponible',
    'Assignment saved and shared with GLaDOS.': 'Mission enregistrée et partagée avec GLaDOS.',
    // Context
    'Inspect GLaDOS’s inference context.': "Inspectez le contexte d'inférence de GLaDOS.", 'Core request': 'Requête du noyau',
    'Central Core · user request': 'Noyau central · demande utilisateur', 'Autonomy Core · autonomous check': "Noyau d'autonomie · vérification autonome",
    'Context source': 'Source du contexte', 'Live preview': 'Aperçu en direct', 'Last submitted request': 'Dernière requête envoyée',
    'Latest preview': 'Dernier aperçu', 'Latest request': 'Dernière requête', 'Refresh': 'Actualiser',
    'Live updates: ON': 'Mises à jour en direct : ON', 'Live updates: OFF': 'Mises à jour en direct : OFF', 'Download JSON': 'Télécharger le JSON',
    'Show all messages': 'Afficher tous les messages', 'Collapse all': 'Tout replier',
    'Messages are numbered in the order sent: reusable instructions, conversation history, live state, then the current input. Live preview shows what a new request would receive; new voice or text is added when it arrives. Last submitted request shows the actual payload. Tools are displayed separately below the messages.':
      "Les messages sont numérotés dans l'ordre d'envoi : consignes réutilisables, historique, état en direct, puis l'entrée actuelle. L'aperçu en direct montre ce que recevrait une nouvelle requête ; la voix ou le texte s'ajoute à l'arrivée. La dernière requête envoyée montre le contenu réel. Les outils sont affichés à part, sous les messages.",
    'Find in context': 'Chercher dans le contexte', 'Search prompts, core reports or conversation…': 'Chercher dans les consignes, rapports ou la conversation…',
    'Open the Neural Buffer to inspect the connected engine.': 'Ouvrez Contexte pour inspecter le moteur connecté.',
    'Available tool catalogue · preview only': 'Catalogue des outils · aperçu seulement', 'Tools offered in this request': 'Outils proposés dans cette requête',
    'This catalogue is not a message sent to GLaDOS. Routing selects the offered subset for the next request.':
      "Ce catalogue n'est pas un message envoyé à GLaDOS. Le routage choisit le sous-ensemble proposé à la prochaine requête.",
    'These definitions were sent in the separate tools field. They have no message number; their placement in the model template depends on the backend.':
      "Ces définitions ont été envoyées dans le champ « tools ». Elles n'ont pas de numéro ; leur place dans le modèle dépend du backend.",
    'Engine disconnected. Displayed context may be stale.': 'Moteur déconnecté. Le contexte affiché peut être périmé.',
    'Context could not be loaded.': "Le contexte n'a pas pu être chargé.", 'Loading context…': 'Chargement du contexte…',
    // Memory
    'Explore what GLaDOS remembers.': 'Explorez ce dont GLaDOS se souvient.', 'Recalled for this conversation': 'Rappelé pour cette conversation',
    'Waiting for a conversation topic.': "En attente d'un sujet de conversation.", 'Saved memory': 'Mémoire enregistrée',
    'Browse saved facts and conversation summaries. Searching here does not change what GLaDOS recalls.':
      'Parcourez les faits et résumés enregistrés. Chercher ici ne change pas ce que GLaDOS se rappelle.',
    'Search memory': 'Chercher dans la mémoire', 'A name, preference or project…': 'Un nom, une préférence ou un projet…',
    'Show': 'Afficher', 'Facts and summaries': 'Faits et résumés', 'Facts': 'Faits', 'Summaries': 'Résumés', 'Search': 'Chercher',
    'Open Test Chamber to browse saved memory.': 'Ouvrez Mémoire pour parcourir les souvenirs.', 'Previous': 'Précédent', 'Next': 'Suivant',
    'No saved memories match this view.': 'Aucun souvenir ne correspond à cette vue.', 'Summary': 'Résumé', 'Fact': 'Fait',
    'Date unavailable': 'Date indisponible', 'Engine disconnected. Displayed memory may be stale.': 'Moteur déconnecté. La mémoire affichée peut être périmée.',
    'Memory could not be loaded.': "La mémoire n'a pas pu être chargée.", 'Recall unavailable.': 'Rappel indisponible.',
    'The Memory Core is paused.': 'Le noyau Mémoire est en pause.', 'No saved facts match the current topic.': 'Aucun fait ne correspond au sujet actuel.',
    'Recall is disabled.': 'Le rappel est désactivé.', 'No saved memories in this view.': 'Aucun souvenir dans cette vue.',
    'Memory could not be read': "La mémoire n'a pas pu être lue", 'Memory change failed': 'La modification de la mémoire a échoué',
    // Settings
    'Interface appearance': "Apparence de l'interface",
    'Dark, Clinical, Terminal and PotatOS share the same live controls. This browser remembers your theme. Open the Presence screen from the top bar for the tablet view.':
      "Sombre, Clinique, Terminal et PotatOS partagent les mêmes commandes. Ce navigateur retient votre thème. Ouvrez l'Écran de présence depuis la barre du haut pour la vue tablette.",
    'Dark': 'Sombre', 'Clinical': 'Clinique', 'Terminal': 'Terminal', 'PotatOS': 'PotatOS',
    'Internet search sources': 'Sources de recherche Internet',
    'News requests read these sites’ pages directly, then search if needed. Other categories guide web searches. Enter one domain or site path per line, up to eight per category. Console changes apply to the next request and save to data/search_settings.yaml. File edits apply after restart.':
      "Les demandes d'actualité lisent directement ces sites, puis cherchent si besoin. Les autres catégories orientent les recherches web. Un domaine ou chemin par ligne, huit maximum par catégorie. Les changements faits ici s'appliquent à la prochaine requête et sont enregistrés dans data/search_settings.yaml. Les modifications du fichier s'appliquent après redémarrage.",
    'Weather': 'Météo', 'News': 'Actualités', 'Reddit communities': 'Communautés Reddit', 'General sources': 'Sources générales',
    'A favorite domain or site path': 'Un domaine ou chemin favori', 'Save favorite sources': 'Enregistrer les sources favorites',
    'Use at most eight sites per category.': 'Huit sites maximum par catégorie.',
    'Favorite search sources saved. They apply to the next search.': 'Sources favorites enregistrées. Elles s’appliquent à la prochaine recherche.',
    'This device as microphone': 'Cet appareil comme micro',
    'Talk to GLaDOS through this browser, for example a phone: its microphone replaces the PC microphone and GLaDOS speaks through it. The PC takes over again when you stop or close the page. While GLaDOS speaks here, this microphone is muted so she does not hear herself. Browsers allow the microphone only over HTTPS.':
      "Parlez à GLaDOS via ce navigateur, par exemple un téléphone : son micro remplace celui du PC et GLaDOS parle à travers lui. Le PC reprend la main quand vous arrêtez ou fermez la page. Pendant que GLaDOS parle ici, ce micro est coupé pour qu'elle ne s'entende pas. Les navigateurs n'autorisent le micro qu'en HTTPS.",
    'Stop using this device': "Ne plus utiliser cet appareil", 'Microphone in use: PC.': 'Micro utilisé : PC.',
    'Remote audio is not available with this audio backend.': "L'audio distant n'est pas disponible avec ce backend audio.",
    'Needs HTTPS: set webapp.tls: true and open the https:// address.': "HTTPS requis : activez webapp.tls: true et ouvrez l'adresse https://.",
    'Another device took over the microphone.': 'Un autre appareil a pris le micro.', 'Microphone permission was refused.': "L'accès au micro a été refusé.",
    'Remote audio failed.': "L'audio distant a échoué.",
    'Consultation lets GLaDOS ask external advisor AIs in the background. The base decides where Central Core runs: auto moves it to the API when this machine is too slow, local and api force it. Needs the API key named in the configuration. Changes apply immediately and save to data/nexus_settings.yaml.':
      "La consultation permet à GLaDOS d'interroger des IA conseillères externes en arrière-plan. La base décide où tourne le noyau central : auto le bascule vers l'API quand cette machine est trop lente, local et api l'imposent. Nécessite la clé API indiquée dans la configuration. Les changements s'appliquent tout de suite et sont enregistrés dans data/nexus_settings.yaml.",
    'Nexus Core unavailable.': 'Noyau Nexus indisponible.', 'Consultation': 'Consultation', 'On': 'Activé', 'Off': 'Désactivé',
    'Base model': 'Modèle de base', 'Auto': 'Auto', 'Local only': 'Local uniquement', 'API only': 'API uniquement',
    'Daily budget (USD)': 'Budget quotidien (USD)', 'Save Nexus settings': 'Enregistrer les réglages Nexus',
    'Choose a daily budget from 0 to 1000 USD.': 'Choisissez un budget quotidien entre 0 et 1000 USD.', 'Nexus settings saved.': 'Réglages Nexus enregistrés.',
    'autonomy, voice input and routing.': 'autonomie, entrée vocale et routage.', 'Create decision list': 'Créer une liste de décision',
    'Facility Sensors & Audio': 'Capteurs et audio du centre',
    'Choose the webcam, microphone and speakers connected to the GLaDOS host. Changes apply immediately for this session.':
      "Choisissez la webcam, le micro et les haut-parleurs branchés sur la machine de GLaDOS. Les changements s'appliquent tout de suite pour cette session.",
    'Webcam input': 'Webcam', 'Audio input': 'Entrée audio', 'Audio output': 'Sortie audio', 'Refresh devices': 'Actualiser les appareils',
    'Open Facility Settings to list connected devices.': 'Ouvrez Réglages pour lister les appareils.',
    'System default': 'Par défaut du système', 'Connect the engine to select host devices.': 'Connectez le moteur pour choisir les appareils.',
    'Listing connected devices…': 'Recherche des appareils…', 'Devices could not be updated.': "Les appareils n'ont pas pu être mis à jour.",
    'Connected devices refreshed.': 'Appareils actualisés.',
    'Motion guides each image analysis delay: a still scene favours the maximum, movement favours the minimum, with some randomness. Default: 2–5 seconds. Enter whole seconds; changes apply immediately and are saved across restarts. The blink follows actual image analysis.':
      "Le mouvement règle le délai entre deux analyses d'image : une scène immobile tend vers le maximum, le mouvement vers le minimum, avec un peu d'aléatoire. Par défaut : 2–5 secondes. Secondes entières ; appliqué tout de suite et conservé après redémarrage. Le clignement suit l'analyse réelle.",
    'Minimum delay (seconds)': 'Délai minimum (secondes)', 'Maximum delay (seconds)': 'Délai maximum (secondes)', 'Save interval range': "Enregistrer l'intervalle",
    'Choose whole seconds from 1 to 60, with minimum no greater than maximum.': 'Secondes entières de 1 à 60, minimum inférieur ou égal au maximum.',
    'Quiet mode': 'Mode silence',
    'Say “go to sleep” or “be quiet” to stop replies and suspend background cores. She keeps listening for “wake up”. Background sounds and unintelligible speech are ignored.':
      "Dites « go to sleep » ou « be quiet » pour arrêter les réponses et suspendre les noyaux de fond. Elle continue d'écouter « wake up ». Les bruits de fond et paroles inintelligibles sont ignorés.",
    'Reviews all core slots and recent conversation during idle pauses. Useful new alerts, completed searches and observed arrivals are passed to Central Core, which speaks in GLaDOS’s normal voice. Other cores run independently; silent checks back off automatically.':
      "Passe en revue les noyaux et la conversation récente pendant les pauses. Les alertes utiles, recherches terminées et arrivées observées sont transmises au noyau central, qui parle avec la voix habituelle de GLaDOS. Les autres noyaux tournent indépendamment ; les vérifications sans résultat s'espacent automatiquement.",
    'Capability routing': 'Routage des capacités',
    'GLaDOS manages routing and parallel replies automatically. Choose which list describes your available actions. Scores express preference among your options, not certainty.':
      "GLaDOS gère automatiquement le routage et les réponses parallèles. Choisissez la liste qui décrit vos actions disponibles. Les scores expriment une préférence entre options, pas une certitude.",
    'Voice input': 'Entrée vocale',
    'Optional transcripts use Gemma to keep a written version of your voice input. Routing, Emotion and replies reuse that text. Available in direct-audio mode; loads no Parakeet.':
      "Les transcriptions optionnelles utilisent Gemma pour garder une version écrite de votre voix. Le routage, l'émotion et les réponses réutilisent ce texte. Disponible en mode audio direct ; ne charge pas Parakeet.",
    'Transcripts': 'Transcriptions', 'Routing tree': 'Arbre de routage',
    'Choose the capability area, then its action. MCP services add a server choice before the tool. Click a decision to expand its choices; teal marks the latest route, yellow marks uncertainty.':
      "Choisissez le domaine, puis l'action. Les services MCP ajoutent le choix du serveur avant l'outil. Cliquez sur une décision pour déplier ses choix ; le turquoise marque la dernière route, le jaune l'incertitude.",
    'Saved. Applies to the next response.': "Enregistré. S'applique à la prochaine réponse.", 'Response instructions': 'Consignes de réponse',
    'Inspector updated.': 'Inspecteur mis à jour.', 'Camera schedule updated.': 'Planification de la caméra mise à jour.',
    'One observation queued.': 'Une observation en file.', 'Reconnect to the engine before making changes.': 'Reconnectez le moteur avant de modifier.',
    'The engine rejected the request.': 'Le moteur a refusé la demande.', 'Engine unavailable': 'Moteur indisponible',
    'Wake GLaDOS': 'Réveiller GLaDOS', 'Put GLaDOS to sleep': 'Endormir GLaDOS',
    'Shut down': 'Arrêter', 'Shut down GLaDOS': 'Arrêter GLaDOS', 'Arrêter': 'Arrêter', 'Arrêter GLaDOS': 'Arrêter GLaDOS',
    'Quiet mode: listening for wake requests.': "Mode silence : en attente d'une demande de réveil.", 'GLaDOS is awake.': 'GLaDOS est réveillée.',
    'Autonomy enabled.': 'Autonomie activée.', 'Autonomy disabled.': 'Autonomie désactivée.', 'Enabled': 'Activé', 'Disabled': 'Désactivé',
    'Camera enabled.': 'Caméra activée.', 'Camera paused and released.': 'Caméra en pause et libérée.',
    'Active routing list saved.': 'Liste de routage active enregistrée.', 'Decision list deleted.': 'Liste de décision supprimée.',
    'Decision list saved.': 'Liste de décision enregistrée.', 'Enter valid JSON': 'Saisissez un JSON valide',
    'No route selected yet.': 'Aucune route choisie pour le moment.', 'No matching choice': 'Aucun choix correspondant',
    'Use audio of 30 seconds or less.': 'Utilisez un audio de 30 secondes maximum.', 'Audio file must be under 10 MB.': 'Le fichier audio doit faire moins de 10 Mo.',
    'Stop recording': "Arrêter l'enregistrement", 'Text mode': 'Mode texte', 'Preview complete. Nothing executed.': 'Aperçu terminé. Rien n’a été exécuté.',
    'Reply': 'Répondre', 'Ignore background speech': 'Ignorer les voix de fond', 'Ask for clarification': 'Demander une précision',
    'Sleep · listening for wake': 'Veille · écoute du réveil', 'Wake · resume replies': 'Réveil · reprise des réponses',
    'Assistant planning': "Planification de l'assistant", 'Configured fallback · no tool selected': 'Repli configuré · aucun outil choisi',
    'Assistant fills arguments': "L'assistant remplit les arguments",
    'Routing decision tree. Click a decision to expand or collapse its choices.': 'Arbre de décision du routage. Cliquez sur une décision pour déplier ou replier ses choix.',
    'Routing is automatic. Private drafts use spare capacity and are released only after routing and Emotion Core approve the reply.':
      "Le routage est automatique. Les brouillons privés utilisent la capacité restante et ne sont diffusés qu'après validation par le routage et le noyau Émotion.",
    'This model handles routing in its replies.': 'Ce modèle gère le routage dans ses réponses.',
    // Diagnostics
    'observability bus stream.': "flux du bus d'observabilité.", 'engine infrastructure, inference capacity, and device health.':
      "infrastructure du moteur, capacité d'inférence et état des appareils.",
    'Hardware & Model Adaptation': 'Matériel et adaptation du modèle', 'Reply latency': 'Latence de réponse',
    'Engine components': 'Composants du moteur', 'Audio': 'Audio', 'Interaction': 'Interaction', 'Vision': 'Vision',
    'MCP servers': 'Serveurs MCP', 'Inference diagnostics': "Diagnostics d'inférence", 'the routing, then conversation': 'le routage, puis la conversation',
    'background inference workers': "exécuteurs d'inférence de fond", 'Worker pool': "Groupe d'exécuteurs", 'Inspector': 'Inspecteur',
    'Last reply': 'Dernière réponse', 'Speech recognition': 'Reconnaissance vocale', 'Model first token': 'Premier jeton du modèle',
    'First voice clip': 'Premier extrait vocal', 'Until playback': "Jusqu'à la lecture", 'No reply measured yet': 'Aucune réponse mesurée',
    'Host Tier': 'Niveau de la machine', 'System RAM': 'RAM système', 'Adaptive Context': 'Contexte adaptatif', 'Max Output': 'Sortie max',
    'CPU Cores / Threads': 'Cœurs / threads CPU', 'Inference Timeout': "Délai d'inférence", 'Hardware telemetry standby': 'Télémétrie matérielle en attente',
    'No engine components reported.': 'Aucun composant signalé.', 'Microphone VAD': 'Détection de voix du micro',
    'Waiting for telemetry.': 'En attente de télémétrie.', 'Waiting for the engine to reconnect.': 'En attente de reconnexion du moteur.',
    'Listening only for a wake request.': "N'écoute qu'une demande de réveil.", 'Voice output active.': 'Voix active.',
    'Microphone voice activity detected.': 'Voix détectée au micro.',
    'No face detected. I\'ll wake when you return or speak.': 'Aucun visage détecté. Je me réveillerai quand vous reviendrez ou parlerez.',
    // Vision
    'Live webcam with face tracking': 'Webcam en direct avec suivi du visage', 'Waiting for the first observation.': 'En attente de la première observation.',
    'No previous observation yet.': 'Aucune observation précédente.', 'Waiting for timestamped observations.': "En attente d'observations horodatées.",
    'Paused schedule · observing once…': 'Planification en pause · observation unique…', 'Paused · camera released.': 'En pause · caméra libérée.',
    'Camera active · selecting fresh, sharp frames.': 'Caméra active · sélection d’images récentes et nettes.', 'Connecting to camera…': 'Connexion à la caméra…',
    'Resume Vision Core': 'Reprendre le noyau Vision', 'Suspend Vision Core': 'Suspendre le noyau Vision', 'No frame yet': 'Aucune image',
    'Ask GLaDOS a specific question to inspect the camera.': 'Posez une question précise à GLaDOS pour inspecter la caméra.',
    'No person detected': 'Personne détectée : aucune', 'Presence unclear': 'Présence incertaine',
    'Person present · face location unavailable': 'Personne présente · position du visage inconnue', 'Face in caption frame': 'Visage dans l’image analysée',
    'Face tracking': 'Suivi du visage', 'Compared with': 'Comparé à', 'First observation': 'Première observation',
    'Observation window': "Fenêtre d'observation", 'Caption rate': "Fréquence d'analyse", 'Webcam rate': 'Fréquence webcam',
    'Face detection': 'Détection du visage', 'Frame selection': "Sélection d'image", 'Focus score': 'Score de netteté',
    'Vision Core is unavailable in this profile.': "Le noyau Vision n'est pas disponible dans ce profil.", 'Camera: unavailable': 'Caméra : indisponible',
    'Suspend Vision Core observations and release the webcam.': 'Suspendre les observations et libérer la webcam.',
    'Resume Vision Core observations.': 'Reprendre les observations.', 'Waiting for an observation.': "En attente d'une observation.", 'Camera off.': 'Caméra éteinte.',
    'No extra response instructions. The base personality and tool instructions still apply.':
      "Aucune consigne supplémentaire. La personnalité de base et les consignes d'outils s'appliquent toujours.",
    'Response instructions are unavailable until the engine connects.': "Les consignes de réponse sont indisponibles tant que le moteur n'est pas connecté.",
    // Demo data
    'Watching the sky; weather briefing queued.': 'Surveille le ciel ; bulletin météo en file.', 'Hydration reminder armed. 45m cooldown.': "Rappel d'hydratation armé. 45 min de délai.",
    'Weather brief': 'Bulletin météo', 'Day agenda': 'Agenda du jour', 'Clash detected: sprint sync vs. dentist at 14:00.': 'Conflit : point sprint et dentiste à 14:00.',
    'Hydration nudge': "Rappel d'hydratation",
    'GLaDOS uses the PC microphone and speakers.': 'GLaDOS utilise le micro et les haut-parleurs du PC.',
    'This device is GLaDOS’s microphone and speaker. The PC microphone is ignored until you stop or close this page.':
      "Cet appareil sert de micro et de haut-parleur à GLaDOS. Le micro du PC est ignoré jusqu'à ce que vous arrêtiez ou fermiez cette page.",
    'Speaking': 'Parle', 'Listening': 'Écoute', 'Processing': 'Réfléchit', 'Sleeping': 'Endormie',
    'Paired devices': 'Appareils appairés',
    'Phones and other computers that opened the pairing link. Revoke one to sign it out; this PC never needs pairing. Run “glados pair” on the PC to show the link again.':
      "Téléphones et autres ordinateurs ayant ouvert le lien d'appairage. Révoquez-en un pour le déconnecter ; ce PC n'a jamais besoin d'appairage. Lancez « glados pair » sur le PC pour réafficher le lien.",
    'Sign out every device': 'Déconnecter tous les appareils', 'Revoke': 'Révoquer', 'this device': 'cet appareil',
    'No paired devices': 'Aucun appareil appairé', 'The console only listens on this machine.': "La console n'écoute que sur cette machine.",
    'Every device was signed out.': 'Tous les appareils ont été déconnectés.', 'Device signed out.': 'Appareil déconnecté.',
    'Working…': 'En cours…', 'Updated.': 'Mis à jour.', 'Download the GLaDOS certificate': 'Téléchargez le certificat GLaDOS',
    'and install it once on a phone to remove the security warning for good. Android: Settings › Security › Encryption & credentials › Install a certificate › CA certificate. iPhone: open the file, install the profile, then enable it under Settings › General › About › Certificate Trust Settings. It only covers local addresses, never public websites.':
      "et installez-le une fois sur le téléphone pour supprimer définitivement l'alerte de sécurité. Android : Paramètres › Sécurité › Chiffrement et identifiants › Installer un certificat › Certificat CA. iPhone : ouvrez le fichier, installez le profil, puis activez-le dans Réglages › Général › Informations › Réglages des certificats. Il ne couvre que les adresses locales, jamais les sites publics.",
    'capacity unknown': 'capacité inconnue', 'Reply latency trend': 'Évolution de la latence de réponse', 'End-of-speech silence': 'Silence de fin de phrase', 'Engine unavailable': 'Moteur indisponible',
    // Rendered by scripts
    'Active list': 'Liste active', 'All choices and descriptions': 'Tous les choix et descriptions', 'Expand all': 'Tout déplier',
    'Overview': "Vue d'ensemble", 'Mermaid source': 'Source Mermaid', 'Download Mermaid': 'Télécharger le Mermaid',
    'Show latest route': 'Afficher la dernière route', 'Test without executing': 'Tester sans exécuter', 'Test': 'Tester',
    'Edit': 'Modifier', 'Delete': 'Supprimer', 'Cancel': 'Annuler', 'Device': 'Appareil', 'Unavailable': 'Indisponible',
    'Selected': 'Sélectionné', 'Unknown': 'Inconnu', 'Capacity': 'Capacité', 'Scene': 'Scène', 'Not rated': 'Non évalué',
    'CONFIDENCE': 'CONFIANCE', 'LOW': 'FAIBLE', 'MEDIUM': 'MOYEN', 'HIGH': 'ÉLEVÉ', 'YES': 'OUI', 'no': 'non', 'hot': 'actif', 'idle': 'inactif',
    'report': 'rapport', 'open': 'ouverte', 'done': 'terminée', 'running': 'en cours', 'active': 'actif', 'monitoring': 'surveillance',
    'online': 'en ligne', 'offline': 'hors ligne', 'background': 'arrière-plan', 'never': 'jamais', 'paused': 'en pause',
    'no schedule': 'pas de planification', 'single step': 'étape unique', 'capability → action': 'capacité → action',
    'input level': "niveau d'entrée", 'inspect core →': 'inspecter le noyau →', 'in inference': 'en inférence',
    'collapse −': 'replier −', 'expand +': 'déplier +', 'Decision': 'Décision', 'Fallback': 'Repli',
    'Ready for a command.': 'Prêt pour une commande.', 'Ready for inference.': "Prêt pour l'inférence.",
    'Ready for requested research': 'Prêt pour une recherche demandée', 'Routing disabled in Settings.': 'Routage désactivé dans les réglages.',
    'Background inference waiting for user response': 'Inférence de fond en attente de votre réponse',
    'Transcribed audio': 'Audio transcrit', 'Direct audio': 'Audio direct', 'Listening gate': "Filtre d'écoute",
    'transcript pending': 'transcription en attente', 'no transcript kept': 'pas de transcription conservée',
    'no response recorded': 'aucune réponse enregistrée', 'no reply selected': 'aucune réponse choisie', 'timing unavailable': 'chronométrage indisponible',
    'text delivered (voice muted)': 'texte affiché (voix coupée)', 'playback requested': 'lecture demandée', 'user response': 'votre réponse',
    'Input accepted; waiting for a recorded response event.': "Entrée acceptée ; en attente d'un événement de réponse.",
    'No context sections match this search.': 'Aucune section du contexte ne correspond à cette recherche.',
    'No core reports yet.': 'Aucun rapport de noyau pour le moment.', 'No core state available.': 'Aucun état de noyau disponible.',
    'Some saved memory could not be read.': "Une partie de la mémoire n'a pas pu être lue.", 'Delete this saved memory?': 'Supprimer ce souvenir ?',
    'Memory text': 'Texte du souvenir', 'Excerpt': 'Extrait', 'The index includes the newest memories within its size limit.':
      "L'index contient les souvenirs les plus récents dans la limite de sa taille.",
    'Operator console ↗': 'Console opérateur ↗', 'Tools · separate from message order': 'Outils · hors ordre des messages',
    'GLaDOS’s modeled mood, from −1 to +1. PAD updates before each reply and directs its tone and expression.':
      "Humeur modélisée de GLaDOS, de −1 à +1. Le PAD se met à jour avant chaque réponse et oriente son ton et son expression.",
    // Core titles and roles reported by the engine
    'Answers your requests and uses tools. Works on tracked tasks when asked.':
      'Répond à vos demandes et utilise les outils. Travaille sur les tâches suivies quand on le lui demande.',
    'Behavior Observer Core': 'Noyau Observateur', 'Behavior Observer': 'Observateur', 'Hacker News Core': 'Noyau Hacker News',
    'Conversation and facility control': 'Conversation et contrôle du centre', 'One-token intent classification': "Classification d'intention en un jeton",
    'Independent attention review and Central Core handoffs': "Revue d'attention indépendante et transferts au noyau central",
    'Recall and context management': 'Rappel et gestion du contexte', 'Requested research and source verification': 'Recherches demandées et vérification des sources',
    'System status and alerts': 'État du système et alertes', 'External advisors and API base model': 'Conseillers externes et modèle de base API',
    'emotional regulation': 'régulation émotionnelle', 'meta supervision': 'méta-supervision', 'news monitor': 'veille actualités', 'weather monitor': 'veille météo',
    'Periodic autonomy ticks': "Cycles d'autonomie périodiques", 'Thread active': 'Thread actif', 'Initializing': 'Initialisation', 'Shutdown': 'Arrêté',
    'No summary yet': 'Pas encore de résumé', 'Autonomy mode changed': "Mode d'autonomie modifié", 'cancelled': 'annulé',
    'Requested microphone capture is disconnected': 'La capture micro demandée est déconnectée',
    // Routing tree steps
    'Speech routing': 'Routage vocal', 'Capability area': 'Domaine de capacité', 'Local system': 'Système local', 'MCP services': 'Services MCP',
    'Memory': 'Mémoire', 'Tasks': 'Tâches', 'Other tools': 'Autres outils', 'Quiet command confirmation': 'Confirmation des commandes de silence',
    // Drawers, decision-list editor and routing tester
    'Status': 'Statut', 'Step': 'Étape', 'Task': 'Tâche', 'Time': 'Heure', 'Title': 'Titre', 'Tool': 'Outil', 'Function': 'Fonction',
    'Input': 'Entrée', 'Inputs': 'Entrées', 'Inference': 'Inférence', 'Lane': 'Voie', 'Model': 'Modèle', 'Motion': 'Mouvement',
    'Delay': 'Délai', 'Elapsed': 'Écoulé', 'Enter': 'Entrée', 'Observed': 'Observé', 'Recall': 'Rappel', 'Report': 'Rapport',
    'Running': 'En cours', 'Schedule': 'Planification', 'Sources': 'Sources', 'Capability': 'Capacité', 'Background': 'Arrière-plan',
    'Voice': 'Voix', 'Core memory': 'Mémoire du noyau', 'Latest output': 'Dernière sortie', 'Latest question': 'Dernière question',
    'Research round': 'Tour de recherche', 'Observation timing': 'Rythme des observations', 'Speech + interaction': 'Parole + interaction',
    'What I see': 'Ce que je vois', 'What changed since the previous image': "Ce qui a changé depuis l'image précédente",
    'Recent events · last four observations': 'Événements récents · quatre dernières observations', 'no visual feed': "pas d'image",
    'none configured': 'aucun configuré', 'Facility settings': 'Réglages', 'Facility tools': 'Outils', 'Routing structure': 'Structure du routage',
    'Run once now': 'Exécuter une fois', 'Run command': 'Lancer la commande', 'Select a tool': 'Choisir un outil',
    'Run once queues work on this core’s worker, including while paused. A current run is allowed to finish.':
      "Exécuter une fois met une tâche en file sur l'exécuteur de ce noyau, même en pause. Une exécution en cours peut se terminer.",
    'Ask GLaDOS to research a topic to start a search. Suspending this core cancels its current research.':
      'Demandez à GLaDOS de rechercher un sujet pour lancer une recherche. Suspendre ce noyau annule la recherche en cours.',
    'Ask GLaDOS to use this capability from the Enrichment Center.': "Demandez à GLaDOS d'utiliser cette capacité depuis l'Accueil.",
    'Uses the tools listed in Facility Tools. Your response instructions apply to this core.':
      "Utilise les outils listés dans Outils. Vos consignes de réponse s'appliquent à ce noyau.",
    'Reviews independent core slots and conversation evidence, then asks Central Core to deliver useful new notifications.':
      'Examine les noyaux indépendants et la conversation, puis demande au noyau central de transmettre les nouvelles notifications utiles.',
    'Searches, reviews source evidence and follows up to resolve gaps. Citations use returned URLs; quoted findings are checked against the source excerpts.':
      'Cherche, examine les sources et approfondit pour combler les lacunes. Les citations utilisent les URL renvoyées ; les passages cités sont vérifiés dans les extraits.',
    'Recalls relevant saved facts into its slot for the current topic. Keeps recent messages intact and merges older context into progressively coarser summaries. Runs quietly; it never initiates speech.':
      "Rappelle les faits pertinents pour le sujet en cours. Garde les messages récents intacts et fusionne l'ancien contexte en résumés de plus en plus condensés. Tourne en silence ; ne prend jamais la parole.",
    'Background processing is unavailable. The Central Core and test assignments remain available.':
      'Le traitement de fond est indisponible. Le noyau central et les tâches restent disponibles.',
    'Choose a fixed read-only task. The assistant uses this same execution slot.': "Choisissez une tâche fixe en lecture seule. L'assistant utilise ce même emplacement d'exécution.",
    'Applied to the next request. These session instructions reset when the engine restarts.':
      "Appliquées à la prochaine requête. Ces consignes de session sont réinitialisées au redémarrage du moteur.",
    'Edit instructions': 'Modifier les consignes', 'Save instructions': 'Enregistrer les consignes', 'Use default': 'Par défaut', 'Use defaults': 'Valeurs par défaut',
    'Ask GLaDOS to work on this': 'Demander à GLaDOS de s’en occuper', 'Assignment or current summary': 'Mission ou résumé actuel',
    'Save assignment': 'Enregistrer la mission', 'Loading task…': 'Chargement de la tâche…', 'Loading latest report and memory…': 'Chargement du dernier rapport et de la mémoire…',
    'Saved in this session’s shared context. Creating an assignment does not start background work.':
      "Enregistrée dans le contexte partagé de la session. Créer une mission ne lance pas de travail de fond.",
    'No assignments.': 'Aucune mission.', 'No assignments yet. Create one here, or ask GLaDOS to save a task.':
      'Aucune mission pour le moment. Créez-en une ici ou demandez à GLaDOS d’enregistrer une tâche.',
    'No cores connected.': 'Aucun noyau connecté.', 'No memory entries yet.': 'Aucun souvenir pour le moment.',
    'No API key set.': 'Aucune clé API.', 'No API key is set, so Nexus stays inactive.': 'Aucune clé API : Nexus reste inactif.',
    'Edit decision lists': 'Modifier les listes de décision', 'New decision list': 'Nouvelle liste de décision', 'Save list': 'Enregistrer la liste',
    'Delete list': 'Supprimer la liste', 'Add option': 'Ajouter une option', 'Delete option': "Supprimer l'option", 'Score options': 'Noter les options',
    'No decision lists. Create one to get started.': 'Aucune liste de décision. Créez-en une pour commencer.',
    'Delete this list and its options? If it is active, GLaDOS switches to another enabled list or restores the default list.':
      'Supprimer cette liste et ses options ? Si elle est active, GLaDOS passe à une autre liste activée ou rétablit la liste par défaut.',
    'Classifier instructions': 'Consignes du classifieur', 'Options · letters assigned automatically': 'Options · lettres attribuées automatiquement',
    'Choose the option that best matches the user’s intent. Quoted and negated commands are not action requests.':
      "Choisissez l'option qui correspond le mieux à l'intention. Les commandes citées ou niées ne sont pas des demandes d'action.",
    'One output token selects an option from your active decision list. Runs on completed turns, before speech or tools.':
      'Un seul jeton de sortie choisit une option de votre liste active. Tourne sur les tours terminés, avant la parole ou les outils.',
    'The active list uses one flat choice.': 'La liste active utilise un choix unique à plat.', 'List enabled': 'Liste activée',
    'How should GLaDOS respond?': 'Comment GLaDOS doit-elle répondre ?', 'Ignore spoken input': "Ignorer l'entrée vocale",
    'Ask a specific clarification': 'Demander une précision', 'Let GLaDOS answer (read-only tools)': 'Laisser GLaDOS répondre (outils en lecture seule)',
    'Automatic from tool': "Automatique selon l'outil", 'Capability, then action': 'Capacité, puis action', 'Single flat choice': 'Choix unique à plat',
    'Ordinary conversation': 'Conversation ordinaire', 'Unclear request': 'Demande peu claire', 'Uncertain decision': 'Décision incertaine',
    'Every routing step passed': 'Toutes les étapes du routage validées', 'Preview only. No tool executes and no speech plays.':
      "Aperçu seulement. Aucun outil n'est exécuté et rien n'est dit.", 'Treat text as spoken input': 'Traiter le texte comme une entrée vocale',
    'Record speech (max 30 s)': 'Enregistrer la voix (30 s max)', 'Or choose audio': 'Ou choisir un audio', 'Clear audio': "Effacer l'audio",
    'Displayed context may be stale.': 'Le contexte affiché peut être périmé.', 'Disconnected': 'Déconnecté',
    'DUMMY DATA — attach to a live engine for real telemetry': 'DONNÉES FICTIVES — connectez un moteur pour la vraie télémétrie',
    'autonomy lane armed (2 workers)': "voie d'autonomie armée (2 exécuteurs)", 'Partly cloudy, 16–19 °C.': 'Partiellement nuageux, 16–19 °C.',
    'Next nudge due in ~27 min.': 'Prochain rappel dans ~27 min.', 'Syncing calendar; found a clash tomorrow 14:00–15:30.':
      'Synchronisation du calendrier ; conflit demain 14:00–15:30.', 'room lighting dimmed': 'éclairage de la pièce tamisé',
    'priority lane worker': 'exécuteur de la voie prioritaire', 'autonomy pool worker': "exécuteur du groupe d'autonomie",
    // Agent panel
    'Agent': 'Agent', 'what GLaDOS does on this computer.': 'ce que GLaDOS fait sur cet ordinateur.',
    'GLaDOS runs commands, edits files and controls the desktop by herself. Blacklisted actions wait here until you approve them, or until you say your confirmation code. Every action is also written to data/agent_audit.jsonl.':
      "GLaDOS lance des commandes, modifie des fichiers et pilote le bureau toute seule. Les actions de la liste noire attendent ici que vous les autorisiez, ou que vous disiez votre code de confirmation. Chaque action est aussi enregistrée dans data/agent_audit.jsonl.",
    'Waiting for approval': 'En attente de validation', 'Background jobs': 'Tâches de fond', 'Recent actions': 'Actions récentes',
    'Notifications: off': 'Notifications : désactivées', 'Notifications: on': 'Notifications : activées',
    'Notifications unavailable': 'Notifications indisponibles', 'No action is waiting for approval.': 'Aucune action en attente de validation.',
    'No background jobs.': 'Aucune tâche de fond.', 'No agent actions yet.': "Aucune action de l'agent pour le moment.",
    'Approve': 'Autoriser', 'Deny': 'Refuser', 'Worker': 'Sous-agent', 'OK': 'OK', 'By': 'Par', 'Details': 'Détails', 'Result': 'Résultat',
    'Approved. Ask GLaDOS to go ahead.': 'Autorisé. Demandez à GLaDOS de continuer.', 'Denied.': 'Refusé.',
    'GLaDOS needs your approval': 'GLaDOS a besoin de votre validation', 'Background job finished': 'Tâche de fond terminée',
    'Self-improvement proposals': "Propositions d'amélioration",
    'Changes GLaDOS prepared to her own code or configuration, in an isolated copy, tested before they reach you. Read the change, approve it above, then ask GLaDOS to apply it. She restarts, and the change is undone automatically if she does not come back.':
      "Modifications que GLaDOS a préparées sur son propre code ou sa configuration, dans une copie isolée, testées avant de vous parvenir. Lisez la modification, autorisez-la plus haut, puis demandez à GLaDOS de l'appliquer. Elle redémarre, et la modification est annulée automatiquement si elle ne revient pas.",
    'No proposals. Ask GLaDOS to improve her code or change a model.': "Aucune proposition. Demandez à GLaDOS d'améliorer son code ou de changer de modèle.",
    'Ready: approve to apply': 'Prête : à autoriser', 'Rejected by tests': 'Refusée par les tests', 'No change': 'Aucun changement',
    'Applying…': 'Application…', 'Applied': 'Appliquée', 'Reverted': 'Annulée', 'Could not apply': "Impossible d'appliquer",
    'Loading the change…': 'Chargement de la modification…',
    // Hardware tier recommendations (core/hardware.py)
    'Ultra high-end workstation. Fits 14B-32B models effortlessly with large context.':
      'Station de travail très haut de gamme. Fait tourner sans effort des modèles 14B-32B avec un grand contexte.',
    'Dedicated GPU / Apple Silicon High Tier. Excellent for 7B-14B models (e.g. Qwen2.5-14B, Llama-3.1-8B).':
      'GPU dédié / Apple Silicon haut de gamme. Excellent pour les modèles 7B-14B (ex. Qwen2.5-14B, Llama-3.1-8B).',
    'Intel Arc / Mid-range GPU. Best suited for 3B-7B models (e.g. Qwen3-4B-Instruct, Gemma4-E4B) with 8k context.':
      'Intel Arc / GPU milieu de gamme. Idéal pour les modèles 3B-7B (ex. Qwen3-4B-Instruct, Gemma4-E4B) avec 8k de contexte.',
    'CPU or low memory system. Recommend compact models (1.5B-3B, e.g. Qwen2.5-3B, Gemma2-2B) with 2k-4k context.':
      'CPU ou peu de mémoire. Modèles compacts recommandés (1.5B-3B, ex. Qwen2.5-3B, Gemma2-2B) avec 2k-4k de contexte.',
  };
  // Strings that embed a live value.
  const FR_PATTERNS = [
    [/^(\d+) in-flight$/, '$1 en cours'], [/^in-flight (\d+)$/, 'en cours $1'], [/^shared processing (.*)$/, 'calcul partagé $1'],
    [/^UP (.*)$/, 'ACTIF $1'], [/^(MOTION|MOVEMENT) (.*)$/, 'MOUVEMENT $2'], [/^Median of last (\d+)$/, 'Médiane des $1 dernières'], [/^Quiet: (ON|OFF|connecting)$/, 'Silence : $1'],
    [/^Autonomy: (ON|OFF|connecting)$/, 'Autonomie : $1'], [/^Camera: (ON|OFF)$/, 'Caméra : $1'],
    [/^Reply started (\d+) ms after the user finished$/, 'Réponse commencée $1 ms après la fin de la phrase'],
    [/^Demo · (.+)$/, (_, state) => 'Démo · ' + (translate(state) ?? state)],
    [/^(\d+) inferences? in progress\.$/, '$1 inférence(s) en cours.'],
    [/^(\d+) \/ (\d+) channels active( · Background inference waiting for user response)?$/,
      (_, used, total, hold) => `${used} / ${total} canaux actifs` + (hold ? ' · Inférence de fond en attente de votre réponse' : '')],
    [/^Engine unavailable · (.*)$/, 'Moteur indisponible · $1'],
    [/^(.+) · before the timer starts$/, '$1 · avant le début du chrono'],
    [/^(.+) · GLaDOS \| Aperture Science$/, (_, view) => `${translate(view) ?? view} · GLaDOS | Aperture Science`],
    [/^(Microphone|Voice|Transcripts): (on|off)$/, (_, what, on) =>
      ({Microphone: 'Micro', Voice: 'Voix', Transcripts: 'Transcriptions'})[what] + ' : ' + (on === 'on' ? 'activé' : 'désactivé')],
    [/^(\d+) active$/, (_, n) => `${n} ${n > 1 ? 'actifs' : 'actif'}`], [/^(\d+ \/ \d+) active$/, '$1 actifs'],
    [/^(\d+) queued$/, '$1 en file'], [/^(\d+) runs$/, '$1 exécutions'], [/^(\d+) configured$/, '$1 configuré(s)'],
    [/^(\d+) choices$/, '$1 choix'], [/^(\d+) saved choices$/, '$1 choix enregistrés'], [/^(\d+) tools$/, '$1 outils'],
    [/^(\d+) tokens$/, '$1 jetons'], [/^(\d+) cores$/, '$1 cœurs'], [/^(\d+) threads$/, '$1 threads'],
    [/^threshold (.+)$/, 'seuil $1'], [/^margin (.+)$/, 'marge $1'], [/^Channel (\d+)$/, 'Canal $1'],
    [/^([\d.]+) GB total$/, '$1 Go au total'], [/^([\d.]+) GB free$/, '$1 Go libres'],
    [/^(\d+) tokens · TTFT (.+)s \(scale (.+)\)$/, '$1 jetons · TTFT $2 s (échelle $3)'],
    [/^([\d.]+[hms]) ago$/, 'il y a $1'], [/^last run (.+)$/, 'dernière exécution $1'],
    [/^(\d+) reserved for interaction$/, "$1 réservé(s) à l'interaction"], [/^(\d+) pending updates$/, '$1 mise(s) à jour en attente'],
    [/^([\d,  ]+) text characters$/, '$1 caractères de texte'], [/^(\d+) available tools \(preview\)$/, '$1 outils disponibles (aperçu)'],
    [/^(\d+) offered tools$/, '$1 outils proposés'], [/^Fixed: (.+)$/, 'Fixe : $1'], [/^Collecting samples \((.+)\)$/, 'Collecte des échantillons ($1)'],
    [/^Current expression: (.+)$/, (_, e) => 'Expression actuelle : ' + (translate(e) ?? e)], [/^Current topic: (.+)$/, 'Sujet actuel : $1'],
    [/^GLaDOS aperture: (.+)$/, 'Œil de GLaDOS : $1'], [/^Vision error: (.+)$/, 'Erreur Vision : $1'], [/^Switching (.+)…$/, 'Changement : $1…'],
    [/^Current device · (.+)$/, 'Appareil actuel · $1'], [/^Disconnected · last known state: (.+)$/, (_, s) => 'Déconnecté · dernier état connu : ' + (translate(s) ?? s)],
    [/^Voice input · (.+)$/, (_, s) => 'Entrée vocale · ' + (translate(s) ?? s)], [/^Last decision: (\w+)(?: — (.+))?$/, (_, o, r) =>
      'Dernière décision : ' + (translate(o) ?? o) + (r ? ' — ' + (translate(r) ?? r) : '')],
    [/^(\w+): (Autonomy mode changed)$/, (_, o, r) => `${translate(o) ?? o} : ${translate(r)}`],
    [/^Showing (\d+)–(\d+) of (\d+) saved memories\.$/, 'Souvenirs $1–$2 sur $3.'],
    [/^(Submitted|Preview updated) (.+?)\. Numbered messages follow the API order; tool definitions are a separate field\. Audio\/image bytes are omitted from inspection\.$/,
      (_, kind, time) => `${kind === 'Submitted' ? 'Envoyée' : 'Aperçu mis à jour'} ${time}. Les messages sont numérotés dans l'ordre de l'API ; les définitions d'outils sont un champ à part. Les octets audio/image sont omis.`],
    [/^RAM usage is (.+)%$/, 'RAM utilisée à $1 %'], [/^Host temperature is (.+)C$/, 'Température de la machine : $1 °C'],
    [/^Model server is (.+)$/, 'Serveur de modèle : $1'], [/^MCP (.+) is disconnected$/, 'MCP $1 déconnecté'],
    [/^Inference queue wait exceeds (.+)s$/, "L'attente de la file d'inférence dépasse $1 s"],
    [/^Context ~(\d+) tokens$/, 'Contexte ~$1 jetons'], [/^recent history intact$/, 'historique récent intact'],
    [/^recall: (\d+) saved facts$/, 'rappel : $1 faits enregistrés'], [/^(\d+) relevant facts$/, '$1 faits pertinents'], [/^(\d+) indexed$/, '$1 indexés'],
    [/^(?:Local first replies (?:now )?estimated at|Local first replies take about) (\d+) s$/, 'Premières réponses locales estimées à $1 s'],
    [/^Central runs on (\S+) \((.+?)\): (.+?)\.( No API key set\.)?( Local first reply ≈ ([\d.]+) s\.)?( Spent today: \$([\d.]+) \/ \$([\d.]+)\.)?( (\d+) consultation\(s\) running\.)?$/,
      (...m) => `Le noyau central tourne en ${m[1]} (${m[2]}) : ${translate(m[3]) ?? m[3]}.` + (m[4] ? ' Aucune clé API.' : '') +
        (m[5] ? ` Première réponse locale ≈ ${m[6]} s.` : '') + (m[7] ? ` Dépensé aujourd'hui : $${m[8]} / $${m[9]}.` : '') +
        (m[10] ? ` ${m[11]} consultation(s) en cours.` : '')],
    [/^Consultation (on|off)$/, (_, on) => 'Consultation ' + (on === 'on' ? 'activée' : 'désactivée')],
    [/^base (\S+) \((.+?)\): (.+)$/, (_, b, m, r) => `base ${b} (${m}) : ${translate(r) ?? r}`],
    [/^local estimate (.+) s$/, 'estimation locale $1 s'], [/^\$(.+)\/\$(.+) today$/, "$$$1/$$$2 aujourd'hui"],
    [/^Daily Nexus budget is exhausted$/, 'Budget Nexus du jour épuisé'], [/^No API key for the base model$/, 'Aucune clé API pour le modèle de base'],
    [/^Base mode is (local|api)$/, 'Mode de base : $1'], [/^Local model server is unreachable$/, 'Serveur de modèle local injoignable'],
    [/^API base failed \((.+)\)$/, "Échec de la base API ($1)"],
    [/^Central Core request: (.+)$/s, 'Requête du noyau central : $1'], [/^Microphone in use: (.+) since (.+)\.$/, 'Micro utilisé : $1 depuis $2.'],
    [/^Motion activity: (\d+)% · next analysis delay: (.+) s$/, 'Activité : $1 % · prochaine analyse dans $2 s'],
    [/^(\d+)% · next delay (.+) s$/, '$1 % · prochain délai $2 s'], [/^Image analysis delay saved: (.+) seconds\.$/, "Délai d'analyse enregistré : $1 secondes."],
    [/^Motion-aware delay (.+) (s|seconds)$/, 'Délai selon le mouvement $1 $2'], [/^Delay (.+) (s|seconds) after completion$/, 'Délai $1 $2 après la fin'],
    [/^Finishing current run$/, "Fin de l'exécution en cours"], [/^idle check in (\d+)s$/, 'vérification dans $1 s'],
    [/^(.+) fps$/, '$1 i/s'], [/^faces (.+) Hz$/, 'visages $1 Hz'], [/^(\d+) ms old$/, 'il y a $1 ms'], [/^(\d+) candidates$/, '$1 candidates'],
    [/^([\d.]+) seconds ready$/, '$1 secondes prêtes'], [/^audio takes precedence over text$/, "l'audio prime sur le texte"],
    [/^(.+)% preference$/, 'préférence $1 %'], [/^lead (.+)$/, 'avance $1'], [/^(\d+) summaries$/, '$1 résumés'],
    [/^Live updates: (ON|OFF)$/, 'Mises à jour en direct : $1'],
    [/^Requested (.+)\. One approval allows one run of this exact action for 3 minutes\.$/,
      'Demandée à $1. Une validation autorise une seule exécution de cette action précise, pendant 3 minutes.'],
    [/^Background job (error|cancelled)$/, (_, s) => 'Tâche de fond ' + (s === 'error' ? 'en échec' : 'annulée')],
  ];
  // Compound labels ("title; detail", "a · b"): translate each part on its own.
  const SEPARATORS = ['; ', ' · '];
  const LANGS = {en: 'English', fr: 'Français'};
  // Free text from GLaDOS, the user, memory and the event log stays as written.
  const SKIP = '#home-wire,#full-wire,#context-sections,#memory-facts,#memory-recall,pre,code,textarea,script,style,[data-no-i18n]';
  const ATTRS = ['placeholder', 'title', 'aria-label'];
  const texts = new WeakMap();  // text node -> {original, shown}
  let lang = 'en';

  const translate = text => {
    if (lang !== 'fr') return null;
    const key = text.replace(/\s+/g, ' ').trim();
    if (!key) return null;
    if (Object.hasOwn(FR, key)) return FR[key];
    for (const [pattern, replacement] of FR_PATTERNS) if (pattern.test(key)) return key.replace(pattern, replacement);
    for (const separator of SEPARATORS) {
      if (!key.includes(separator)) continue;
      const parts = key.split(separator), shown = parts.map(part => translate(part) ?? part);
      if (shown.some((part, i) => part !== parts[i])) return shown.join(separator);
    }
    return null;
  };
  const skipped = node => {
    const element = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
    return !element || !!element.closest(SKIP);
  };
  const textNode = node => {
    if (skipped(node)) return;
    const seen = texts.get(node);
    if (seen && node.data === seen.shown) return;  // our own write
    const original = node.data, translated = translate(original);
    if (translated === null) { texts.delete(node); return; }
    const shown = original.replace(original.trim(), () => translated);
    texts.set(node, {original, shown});
    node.data = shown;
  };
  const attributes = element => {
    if (skipped(element)) return;
    for (const name of ATTRS) {
      if (!element.hasAttribute(name)) continue;
      const stash = 'data-i18n-' + name, value = element.getAttribute(name);
      if (element.getAttribute(stash + '-shown') === value) continue;
      const translated = translate(value);
      if (translated === null) { element.removeAttribute(stash); continue; }
      element.setAttribute(stash, value);
      element.setAttribute(stash + '-shown', translated);
      element.setAttribute(name, translated);
    }
  };
  const walk = root => {
    if (root.nodeType === Node.TEXT_NODE) return textNode(root);
    if (root.nodeType !== Node.ELEMENT_NODE) return;
    attributes(root);
    const tree = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
    for (let node = tree.nextNode(); node; node = tree.nextNode()) node.nodeType === Node.TEXT_NODE ? textNode(node) : attributes(node);
  };
  const restore = () => {
    const tree = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
    for (let node = tree.nextNode(); node; node = tree.nextNode()) {
      if (node.nodeType === Node.TEXT_NODE) {
        const seen = texts.get(node);
        if (seen && node.data === seen.shown) node.data = seen.original;
        texts.delete(node);
      } else for (const name of ATTRS) {
        const stash = 'data-i18n-' + name;
        if (node.hasAttribute(stash) && node.getAttribute(name) === node.getAttribute(stash + '-shown')) node.setAttribute(name, node.getAttribute(stash));
        node.removeAttribute(stash); node.removeAttribute(stash + '-shown');
      }
    }
  };
  const apply = next => {
    lang = LANGS[next] ? next : 'en';
    document.documentElement.lang = lang;
    try { localStorage.setItem('glados-lang', lang); } catch { /* private mode */ }
    lang === 'fr' ? (walk(document.body), document.querySelector('title') && walk(document.querySelector('title'))) : restore();
  };
  const picker = () => {
    const themes = document.querySelector('.theme-label');
    if (!themes || document.getElementById('lang-choice')) return;
    const label = document.createElement('label');
    label.className = 'theme-label';
    const select = document.createElement('select');
    select.id = 'lang-choice';
    select.setAttribute('aria-label', 'Language');
    for (const [code, name] of Object.entries(LANGS)) select.add(new Option(name, code, false, code === lang));
    select.onchange = () => apply(select.value);
    label.append(document.createTextNode('Language '), select);
    themes.after(label);
  };

  let saved = null;
  try { saved = localStorage.getItem('glados-lang'); } catch { /* private mode */ }
  lang = saved || ((navigator.language || '').slice(0, 2) === 'fr' ? 'fr' : 'en');
  window.GladosI18n = {apply, translate: text => translate(text) ?? text, get lang() { return lang; }, dictionary: {fr: FR}};
  const start = () => {
    picker();
    apply(lang);
    const observer = new MutationObserver(records => {
      if (lang !== 'fr') return;
      for (const record of records) {
        if (record.type === 'characterData') textNode(record.target);
        else if (record.type === 'attributes') attributes(record.target);
        else record.addedNodes.forEach(walk);
      }
    });
    observer.observe(document.body, {subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRS});
    // The tab title follows the current view.
    const title = document.querySelector('title');
    if (title) { walk(title); observer.observe(title, {subtree: true, childList: true, characterData: true}); }
  };
  document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', start) : start();
})();
