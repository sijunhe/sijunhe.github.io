---
layout: article
title: "The Effective Engineer"
subtitle: "How to Leverage Your Efforts In Software Engineering to Make a Disproportionate and Meaningful Impact"
tags: reading-notes book non-technical
---
I am always skeptical about productivity books since they can easily come off as very theoretical and of little use for my work. After recommendations from several friends, I decided to read [The Effective Engineer](https://www.amazon.com/Effective-Engineer-Engineering-Disproportionate-Meaningful/dp/0996128107) and found it to be a **highly leveraged** use of my time! As a relatively new engineer, I can already resonate deeply with what Lau wrote. I also started to adopt some of the techniques Lau recommended and have seen good results. Here is my reading notes for [The Effective Engineer](https://www.amazon.com/Effective-Engineer-Engineering-Disproportionate-Meaningful/dp/0996128107).
<!--more-->

# 1 Adopt the Right Mindsets

## 1.1 Focus on High-Leverage Activities
- Leverage is defined as 
$$\text{Leverage} = \frac{\text{Impact Produced}}{\text{Time Invested}}$$
- Example of high leverage activities **mentoring new engineers**. A new hire may spend 1% of the total time being mentored, but it can have an outsized infludnece on the productivity and effectiveness of the other 99% of work hours.
- As engineers, we can always ask ourselves about any activity we are working on:
	- How can I complete this activity in a shorter amount of time?
	- How can I increase the value produced by this activity?
	- Is there something else that I could spend my time on that would produce more value?
- Many high-leverage activities require consistent applications of effort over long time periods to achieve impacts. An example is how Facebook built and maintain a strong hiring culture to attrack strong people. 

## 1.2 Optimize for Learning
- **Adopt a growth mindset**. Believe that I can cultivate and grow their intelligence and skills through effort. View challenges and failures as opportunities to learn.
- **Invest in your learning rate**. Learning compunds like interest. The earlier the compounding starts, the sooner you hit the region of rapid grwoth and reap its benefits. Optimize for learning over profitability, particularly early in your career.
- **Find work environments that can sustain your growth**. Find out what opportunites a company provide for onboarding and mentoring, how transparent they are internally, how fast they move, what your prospective co-wrokers are like, and how much autonomy you'll have.
- **Learn at work**. Work with and learn from your best co-workers. Dive into internal learning resource. Look into classes or books that your workplace might be willing to subsidize.
- **Learn outside of work**. Challenge yourself to become better by just 1% a day, and not limited to engineering skills
	- Read books
	- Attend talks and meetups
	- Build and maintain network of relationships
	- Read and follow blogs
	- Write blogs
	- Tinker on side projects

## 1.3 Prioritize Regularly

![Importance_Urgency_Matrix](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post12/importance_urgency_quadrant.gif)

- **Write down and track To-Dos**. Human brain is optimized for processing and not for storage, so treat your brain as a processor instad of a memory bank. Spend your mental energy on prioritizing and processing the tasks instead of remembering them. (*I have successfully persuaded myself and my team to adopt [Asana](https://app.asana.com/)*)
- **Work on both what directly leads to value and the Important & Non-Urgent**. Along with prioritizing the activities that directly produce value, we also need to prioritize the investments that increase our ability to deliver more value in the future.
- **Protect your Maker's schedule**. Preserve larger blocks of focused time in the schedule to maintain the *flow*. Learn to say no to unimportant activities. 
- **Reduce context switching**. Limit the number of ongoing projects so that you don't spend your cognitive energy actively juggling tasks.
- **Make a routine of prioritization**. Make prioritization part of your daily routine and it will help you to focus on and complete your highest-leverage activities.

# 2 Execute, Execute, Execute

## 2.1 Invest in Iteration Speed

- **Move fast to learn fast**. The faster you can iterate, the more you can learn. Conversely, when you move too slowly trying to avoid mistakes, you lose opportunities.
- **Invest in time-saving tools**. Time-saving tools pays-off large dividends. Faster compile times, faster deployment cyrcles and faster turnaround times for development all provide time-saving benefits that compound the more you use them.
- **Shorten debugging and validation loops**. Create tight testing loops so we can debug faster and iterate quicker.
- **Master fundamentals of your craft**. Get very comfortable with your dev environment (IDE) by learning and defining shortcuts. Use more keyboard than mouse. Learn UNIX shell commands and how to effectively pipe them together.
- **Think about your non-engineering bottlenecks**. Find out where the biggest bottlenecks in your iteration cycle are, whether in engineering, cross-team dependencies, approvals from decision-makers, or organizational processes. Then, work to optimize them.

## 2.2 Measure What You Want to Improve

- **Measure your progress**. If you can't measure it, you can't improve it. Good metrics accomplish a number of goals, ranging from driveing forward progress to measuring effectiveness over time.
- **Pick the right metric**. Different metrics incentivize different behaviors. Example here includes Google using "long click-through rate" to track search quality, companies use 95th/99th percentile of response times instead of average response time to hunt down the worest-case behavior in the system.
- **Instrument your system**. Measure and instrument as much as possible to ensure you are not flying blind.
- **Internalize useful numbers**. Memorize numbers that can benchmark your progress or help wit back-of-the-envelope calculations. Like the [table of common latency numbers](https://gist.github.com/jboner/2841832)
- **Prioritize data integrity**. Having bad data is wrose than having no data because you will make the wrong decisions thinking that you are right. Here are some strategies you can use to increase confidence in your data integrity:
	- Log data liberally, in case it turns out to be useful later on.
	- Build tools to iterate on data accuracy sooner
	- Write end-too-end integration tests to validate your entire analytics pipeline
	- Examine collected data sooner.
	- Cross-validate data accuracy by computing the same metric in mulitple ways.
	- When a number does look off, dig in to it early.

## 2.3 Validate Your Ideas Early and Often

- **Find low-effort ways to validate your work**. Invest a little effort (i.e. minimum viable product) to figure out if the rest of your plan is worth doing. 
- **Continously validate your product hypotheses**. Use A/B testing to incremtnally develop a product and identify what works and doesn't work. This way, you increase the probability that your efforts are aligned with what users actually want.
- **Find way of soliciting feedback when working on solo projects**. Being the one-man team runs the huge risk of overlooking something that , if spotted early, could save you lots of wasted effort. Some strategies are:
	- Ask to bounce ideas off your teammates.
	- Send out a design document before devoting your energy to your code.
	- Structure ongoing projects so that there is some shared context with your teammates.
	- Solicit buy-in before investing too much time.

## 2.4 Improve Your Project Estimation Skills

- **Incorporate estimates into project planning**. Use the estimates to inform project planning, not the other way around. Some concrete strategies that provide more accurate estimates are:
	- Decompose the project into granular tasks
	- Think of estimates as probability distributions, not best-case scenarios
	- Let the person doing the actual task make the estimate
	- Beware of anchoring bias. Avoid committing to an initial number before actually outlining the tasks involved.
	- Use timeboxing to constrain tasks that can grow in scope
- **Budget for the unknown**. Take into account competing work obligations, holidays, illnesses, etc. The longer a project, the higher the probability that some of these will occur.
- **Define measureable milestones**. Define specific goals to reduce risk and efficiently allocate time, and outline milestones to track progress. This allows us to build alignment around what can tasks can be deferred and decreases the chance that a project inadvertently grows in scope.
- **Do the riskiest task first**. Instead of giving yourself the illusion of progress by focusing first on what is easy to do, tackle the rickiest task first to reduce variance in your estimates and explore the unknown early on.
- **Don't sprint in the middle of a marathon**. Don't sprint just because you are hebind. Work overtime only if you are confident that it will enable you to finish on time.

# 3 Build Long-Term Value

## 3.1 Balance Quality with Pragmatism

- **Establish a culture of reviewing code**. Code reviews facilitate positive modeling of good coding practices. Find the right balance between code quality and development speed.
- **Manage complexity through abstraction**. MapReduce illustrates how the right abstraction can dramatically amplify an engineer's output. Make sure you have complete information about use cases before building abstractions, or you will ned up with something clunky and unusable.
- **Scale code quality with automated testing**. Automated testing provide a scalable way of managing a growing codebase with a large team without constantly breaking the build or the product.
- **Manage your technical debt**. Incur technical debt when it's necessary to get things done for a deadline, but to pay off that debt periodically. Only repay the debt with the highest leverage.

## 3.2 Minimize Operational Burden

- **Embrace operational simplicity**. Simple solutions impose a lower operational burden because they are easier to understand, maintain, and modify. Always ask "what is the simplest solution that can get the job done while also reducing our future operational burden"?
- **Build systems to fail fast**. Slow failing systems muddy the sources of code errors, making it difficult to debug. Building systems to fail fast helps reduce the time you spend maintaining and debugging software by surfacing problematic issues sooner and more directly.
- **Automate Mechanical Tasks**. Every time you do something that a machine can do, ask yourself whether it's worthwhile to automate it.
- **Aim for idempotence**. An *idempotent* process produces the same results regardless of whether it's run once or multiple times. Idempotence make it easier for you to retry actions in the face of failure.
- **Plan and practice failure modes**. Building confidence in your ability to recover lets you proceed more boldly.

## 3.3 Invest in Your Team's Growth

- **Help the peopel around you be successful**. The higher you climb up the engineering ladder, the more your effectiveness will be measured not by your individual constributions, but by you rimpact on the people around you. "If you are a staff engineer, you make a whole team better. You are a principal engineer if you are making the whole company better. And you are distinguished if you are improving the industry".
- **Make hiring a priority**. Keep a high hiring bar and play an active role in growing your team.
- **Build shared ownership of code**. Increase your bus factor to be greater than one so that you are not a bottleneck for development. This gives you the flexibility to focus on other high-leverage activities.
- **Document collective wisdom**. Reflect on projects with team member, learn what worked and what didn't work, and document and share the lessions so that valuable wisdom doesn't get lost.
