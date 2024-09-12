grid_problem_template = '''
(define (problem maze) (:domain maze)
(:objects
#add_locationsplayer-1 - player
)
(:init 
#add_player_location#add_goal_location#add_clear_locations#add_location_connectivity
(oriented-right player-1)
)
#add_goal_str)
    
'''

''' (:action wait
        :parameters (?p - player ?where - location)
        :precondition (at ?p ?where)
        :effect (at ?p ?where)
)'''

grid_domain_file = '''

(define (domain maze)
    (:requirements :strips :typing )
    (:types player location)
    (:predicates 
        (move-dir-up ?v0 - location ?v1 - location)
        (move-dir-down ?v0 - location ?v1 - location)
        (move-dir-left ?v0 - location ?v1 - location)
        (move-dir-right ?v0 - location ?v1 - location)
        (clear ?v0 - location)
        (at ?v0 - player ?v1 - location)
        (oriented-up ?v0 - player)
        (oriented-down ?v0 - player)
        (oriented-left ?v0 - player)
        (oriented-right ?v0 - player)
        (is-goal ?v0 - location)
    )
   
    (:action move-up
            :parameters (?p - player ?from - location ?to - location)
            :precondition (and (at ?p ?from)
                    (clear ?to)
                    (move-dir-up ?from ?to))
            :effect (and
                    (not (at ?p ?from))
                    (not (clear ?to))
                    (at ?p ?to)
                    (clear ?from)
                    (not (oriented-down ?p))
                    (not (oriented-left ?p))
                    (not (oriented-right ?p))
                    (oriented-up ?p))
    )

    (:action move-down
            :parameters (?p - player ?from - location ?to - location)
            :precondition (and (at ?p ?from)
                    (clear ?to)
                    (move-dir-down ?from ?to))
            :effect (and
                    (not (at ?p ?from))
                    (not (clear ?to))
                    (at ?p ?to)
                    (clear ?from)
                    (not (oriented-up ?p))
                    (not (oriented-left ?p))
                    (not (oriented-right ?p))
                    (oriented-down ?p))
    )

    (:action move-left
            :parameters (?p - player ?from - location ?to - location)
            :precondition (and (at ?p ?from)
                    (clear ?to)
                    (move-dir-left ?from ?to))
            :effect (and
                    (not (at ?p ?from))
                    (not (clear ?to))
                    (at ?p ?to)
                    (clear ?from)
                    (not (oriented-down ?p))
                    (not (oriented-up ?p))
                    (not (oriented-right ?p))
                    (oriented-left ?p))
    )


    (:action move-right
            :parameters (?p - player ?from - location ?to - location)
            :precondition (and (at ?p ?from)
                    (clear ?to)
                    (move-dir-right ?from ?to))
            :effect (and
                    (not (at ?p ?from))
                    (not (clear ?to))
                    (at ?p ?to)
                    (clear ?from)
                    (not (oriented-down ?p))
                    (not (oriented-left ?p))
                    (not (oriented-up ?p))
                    (oriented-right ?p))
    )

)

'''
'''Example Objects:

apn1 - airplane
 apt1 apt2 - airport
 pos2 pos1 - location
 cit2 cit1 - city
 tru2 tru1 - truck
 obj23 obj22 obj21 obj13 obj12 obj11 - package
 
Example Init:
(at apn1 apt2) (at tru1 pos1) (at obj11 pos1)
 (at obj12 pos1) (at obj13 pos1) (at tru2 pos2) (at obj21 pos2) (at obj22 pos2)
 (at obj23 pos2) (in-city pos1 cit1) (in-city apt1 cit1) (in-city pos2 cit2)
 (in-city apt2 cit2)

Example Goal:
(and (at obj11 apt1) (at obj23 pos1) (at obj13 apt1) (at obj21 pos1))

 '''

logistics_problem_template = '''
(define (problem logistics-4-0)
(:domain logistics)
(:objects
#add_objects
 )

(:init #add_init)

(:goal #add_goal )
)
'''


logistics_domain_file = ''';; logistics domain Typed version.
;;

(define (domain logistics)
  (:requirements :strips :typing) 
  (:types truck
          airplane - vehicle
          package
          vehicle - physobj
          airport
          location - place
          city
          place 
          physobj - object)
  
  (:predicates 	(in-city ?loc - place ?city - city)
		(at ?obj - physobj ?loc - place)
		(in ?pkg - package ?veh - vehicle))
  
(:action LOAD-TRUCK
   :parameters    (?pkg - package ?truck - truck ?loc - place)
   :precondition  (and (at ?truck ?loc) (at ?pkg ?loc))
   :effect        (and (not (at ?pkg ?loc)) (in ?pkg ?truck)))

(:action LOAD-AIRPLANE
  :parameters   (?pkg - package ?airplane - airplane ?loc - place)
  :precondition (and (at ?pkg ?loc) (at ?airplane ?loc))
  :effect       (and (not (at ?pkg ?loc)) (in ?pkg ?airplane)))

(:action UNLOAD-TRUCK
  :parameters   (?pkg - package ?truck - truck ?loc - place)
  :precondition (and (at ?truck ?loc) (in ?pkg ?truck))
  :effect       (and (not (in ?pkg ?truck)) (at ?pkg ?loc)))

(:action UNLOAD-AIRPLANE
  :parameters    (?pkg - package ?airplane - airplane ?loc - place)
  :precondition  (and (in ?pkg ?airplane) (at ?airplane ?loc))
  :effect        (and (not (in ?pkg ?airplane)) (at ?pkg ?loc)))

(:action DRIVE-TRUCK
  :parameters (?truck - truck ?loc-from - place ?loc-to - place ?city - city)
  :precondition
   (and (at ?truck ?loc-from) (in-city ?loc-from ?city) (in-city ?loc-to ?city))
  :effect
   (and (not (at ?truck ?loc-from)) (at ?truck ?loc-to)))

(:action FLY-AIRPLANE
  :parameters (?airplane - airplane ?loc-from - airport ?loc-to - airport)
  :precondition
   (at ?airplane ?loc-from)
  :effect
   (and (not (at ?airplane ?loc-from)) (at ?airplane ?loc-to)))
)'''
